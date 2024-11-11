using UnityEngine;
using Mujoco;
using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

public class IK : MonoBehaviour
{
    // List of MjActuator components that will be used to control the joints.
    public MjActuator[] actuators;

    // Target transform that the IK will try to reach.
    public Transform target;

    // The site that will move to reach the target.
    public MjSite site;

    // Parameters for the IK algorithm.
    public double tolerance = 1e-14;
    public double rotationWeight = 1.0;
    public double regularizationThreshold = 0.1;
    public double regularizationStrength = 3e-2;
    public double maxUpdateNorm = 2.0;
    public double progressThreshold = 20.0;
    public int maxSteps = 100;

    private MjScene _scene;
    private unsafe MujocoLib.mjModel_* _model;
    private unsafe MujocoLib.mjData_* _data;
    private int _siteId;
    private int[] _dofIndices;

    // Preallocated arrays and vectors to minimize allocations.
    private double[] targetPos = new double[3];
    private double[] targetQuat = new double[4];
    private double[] siteXPos = new double[3];
    private double[] siteXMat = new double[9];
    private double[] siteXQuat = new double[4];
    private double[] negSiteXQuat = new double[4];
    private double[] errRotQuat = new double[4];
    private double[] errRot = new double[3];
    private double[] errPos = new double[3];
    private double[] errArray = new double[6];
    private double[] jacp;
    private double[] jacr;
    private double[] jac;
    private double[] updateNV;
    private Vector<double> deltaVector;
    private Matrix<double> jacJoints;
    private Matrix<double> hessApprox;
    private Vector<double> jointDelta;
    private Vector<double> dq;
    private double[] qpos;

    // Variables for the IK loop.
    private int nv;
    private int nq;

    void Start()
    {
        // Get the MuJoCo scene, model, and data.
        _scene = MjScene.Instance;
        if (_scene == null)
        {
            throw new Exception("MuJoCo Scene not found");
        }

        unsafe
        {
            _model = _scene.Model;
            _data = _scene.Data;

            // Get the site ID.
            _siteId = MujocoLib.mj_name2id(_model, (int)MujocoLib.mjtObj.mjOBJ_SITE, site.MujocoName);
            if (_siteId == -1)
            {
                throw new Exception($"Site {site.MujocoName} not found");
            }

            nv = _model->nv;
            nq = _model->nq;

            // Allocate arrays based on model sizes.
            jacp = new double[3 * nv];
            jacr = new double[3 * nv];
            jac = new double[6 * nv];
            updateNV = new double[nv];
            qpos = new double[nq];
        }

        // Get the DOF indices for the actuators controlling the joints.
        _dofIndices = GetDofIndices(actuators);

        // Preallocate matrices and vectors for Math.NET Numerics.
        int errDim = 6;
        int numJoints = _dofIndices.Length;

        jacJoints = Matrix<double>.Build.Dense(errDim, numJoints);
        hessApprox = Matrix<double>.Build.Dense(numJoints, numJoints);
        jointDelta = Vector<double>.Build.Dense(numJoints);
        dq = Vector<double>.Build.Dense(numJoints);
        deltaVector = Vector<double>.Build.Dense(errDim);
    }

    void Update()
    {
        unsafe
        {
            // Create a copy of mjData
            MujocoLib.mjData_* dataCopy = MujocoLib.mj_makeData(_model);

            // Copy the current data into dataCopy
            MujocoLib.mj_copyData(dataCopy, _model, _data);

            // Perform inverse kinematics on the copied data
            IKResult result = QPosFromSitePose(_model, dataCopy);

            if (!result.success)
            {
                Debug.LogWarning($"IK did not converge: error norm = {result.errNorm}");
            }

            // Free the copied data to prevent memory leaks
            MujocoLib.mj_deleteData(dataCopy);

            // Use the actuators to drive the joints toward the IK result
            SetActuatorsToIKResult(result.qpos);
        }
    }

    public struct IKResult
    {
        public double[] qpos;
        public double errNorm;
        public int steps;
        public bool success;
    }

    unsafe IKResult QPosFromSitePose(MujocoLib.mjModel_* model, MujocoLib.mjData_* data)
    {
        bool success = false;
        int steps = 0;
        double errNorm = 0.0;

        // Target position and orientation.
        Vector3 targetPosition = target.position;
        targetPos[0] = targetPosition.x;
        targetPos[1] = targetPosition.z;
        targetPos[2] = targetPosition.y;

        Quaternion unityQuat = target.rotation;
        ConvertUnityQuatToMujoco(unityQuat, targetQuat);

        for (steps = 0; steps < maxSteps; steps++)
        {
            errNorm = 0.0;

            // Perform forward kinematics.
            MujocoLib.mj_fwdPosition(model, data);

            // Get the current site position and orientation.
            for (int i = 0; i < 3; i++)
            {
                siteXPos[i] = data->site_xpos[_siteId * 3 + i];
            }
            for (int i = 0; i < 9; i++)
            {
                siteXMat[i] = data->site_xmat[_siteId * 9 + i];
            }

            // Compute positional error.
            for (int i = 0; i < 3; i++)
            {
                errPos[i] = targetPos[i] - siteXPos[i];
            }
            errNorm += Norm(errPos);

            // Compute rotational error.
            fixed (double* siteXQuatPtr = siteXQuat)
            fixed (double* siteXMatPtr = siteXMat)
            {
                MujocoLib.mju_mat2Quat(siteXQuatPtr, siteXMatPtr);
            }
            fixed (double* negSiteXQuatPtr = negSiteXQuat)
            fixed (double* siteXQuatPtr = siteXQuat)
            {
                MujocoLib.mju_negQuat(negSiteXQuatPtr, siteXQuatPtr);
            }

            fixed (double* errRotQuatPtr = errRotQuat)
            fixed (double* negSiteXQuatPtr = negSiteXQuat)
            fixed (double* targetQuatPtr = targetQuat)
            {
                MujocoLib.mju_mulQuat(errRotQuatPtr, negSiteXQuatPtr, targetQuatPtr);
            }

            fixed (double* errRotPtr = errRot)
            fixed (double* errRotQuatPtr = errRotQuat)
            {
                MujocoLib.mju_quat2Vel(errRotPtr, errRotQuatPtr, 1);
            }

            errNorm += Norm(errRot) * rotationWeight;

            // Check for convergence.
            if (errNorm < tolerance)
            {
                success = true;
                break;
            }

            // Compute the Jacobian.
            fixed (double* jacpPtr = jacp)
            fixed (double* jacrPtr = jacr)
            {
                MujocoLib.mj_jacSite(model, data, jacpPtr, jacrPtr, _siteId);
            }

            Array.Copy(jacp, 0, jac, 0, jacp.Length);
            Array.Copy(jacr, 0, jac, jacp.Length, jacr.Length);

            Array.Copy(errPos, 0, errArray, 0, errPos.Length);
            Array.Copy(errRot, 0, errArray, errPos.Length, errRot.Length);

            ExtractJacobianColumns(jac, _dofIndices, nv, jacJoints);

            for (int i = 0; i < errArray.Length; i++)
            {
                deltaVector[i] = errArray[i];
            }

            NullspaceMethod(jacJoints, deltaVector, regularizationStrength, hessApprox, jointDelta, dq);

            double updateNorm = dq.L2Norm();

            if (updateNorm > maxUpdateNorm)
            {
                double scale = maxUpdateNorm / updateNorm;
                dq.Multiply(scale, result: dq);
            }

            Array.Clear(updateNV, 0, updateNV.Length);

            for (int i = 0; i < _dofIndices.Length; i++)
            {
                updateNV[_dofIndices[i]] = dq[i];
            }

            fixed (double* updateNVPtr = updateNV)
            {
                MujocoLib.mj_integratePos(model, data->qpos, updateNVPtr, 1);
            }

            MujocoLib.mj_fwdPosition(model, data);
        }

        for (int i = 0; i < nq; i++)
        {
            qpos[i] = data->qpos[i];
        }

        return new IKResult
        {
            qpos = qpos,
            errNorm = errNorm,
            steps = steps,
            success = success
        };
    }

    void SetActuatorsToIKResult(double[] targetQpos)
    {
        unsafe
        {
            for (int i = 0; i < actuators.Length; i++)
            {
                var actuator = actuators[i];
                var joint = actuator.Joint;
                if (joint == null)
                {
                    continue;
                }

                int jointId = MujocoLib.mj_name2id(_model, (int)MujocoLib.mjtObj.mjOBJ_JOINT, joint.MujocoName);
                if (jointId == -1)
                {
                    Debug.LogWarning($"Joint {joint.MujocoName} not found");
                    continue;
                }

                int dofAdr = _model->jnt_dofadr[jointId];

                double currentQpos = _data->qpos[dofAdr];
                double desiredQpos = targetQpos[dofAdr];

                double positionError = desiredQpos - currentQpos;

                double gain = 100.0; // Adjust this gain as needed
                double controlInput = gain * positionError;

                _data->ctrl[actuator.MujocoId] = desiredQpos;
                actuator.Control = (float)desiredQpos;
            }
        }
    }

    int[] GetDofIndices(MjActuator[] actuators)
    {
        unsafe
        {
            List<int> dofIndices = new List<int>();

            foreach (var actuator in actuators)
            {
                var joint = actuator.Joint;
                if (joint == null)
                {
                    continue;
                }

                int jointId = MujocoLib.mj_name2id(_model, (int)MujocoLib.mjtObj.mjOBJ_JOINT, joint.MujocoName);
                if (jointId == -1)
                {
                    Debug.LogWarning($"Joint {joint.MujocoName} not found");
                    continue;
                }

                int dofAdr = _model->jnt_dofadr[jointId];
                int dofNum = GetJointDofNum(_model->jnt_type[jointId]);

                for (int i = 0; i < dofNum; i++)
                {
                    dofIndices.Add(dofAdr + i);
                }
            }

            return dofIndices.ToArray();
        }
    }

    int GetJointDofNum(int jointType)
    {
        switch (jointType)
        {
            case (int)MujocoLib.mjtJoint.mjJNT_FREE:
                return 6;
            case (int)MujocoLib.mjtJoint.mjJNT_BALL:
                return 3;
            case (int)MujocoLib.mjtJoint.mjJNT_SLIDE:
            case (int)MujocoLib.mjtJoint.mjJNT_HINGE:
                return 1;
            default:
                return 0;
        }
    }

    void ExtractJacobianColumns(double[] jac, int[] dofIndices, int nv, Matrix<double> jacJoints)
    {
        int errDim = jac.Length / nv;
        int numJoints = dofIndices.Length;

        for (int i = 0; i < errDim; i++)
        {
            for (int j = 0; j < numJoints; j++)
            {
                jacJoints[i, j] = jac[i * nv + dofIndices[j]];
            }
        }
    }

    void NullspaceMethod(Matrix<double> jacJoints, Vector<double> deltaVector, double regularizationStrength,
                         Matrix<double> hessApprox, Vector<double> jointDelta, Vector<double> dq)
    {
        int n = jacJoints.ColumnCount;

        jacJoints.TransposeThisAndMultiply(jacJoints, hessApprox);

        if (regularizationStrength > 0)
        {
            for (int i = 0; i < n; i++)
            {
                hessApprox[i, i] += regularizationStrength;
            }
        }

        jacJoints.TransposeThisAndMultiply(deltaVector, jointDelta);

        try
        {
            hessApprox.Solve(jointDelta, result: dq);
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to solve linear system: {e.Message}");
            dq.Clear();
        }
    }

    double Norm(double[] vec)
    {
        double sum = 0.0;
        for (int i = 0; i < vec.Length; i++)
        {
            sum += vec[i] * vec[i];
        }
        return Math.Sqrt(sum);
    }

    void ConvertUnityQuatToMujoco(Quaternion unityQuat, double[] targetQuat)
    {
        Quaternion swappedQuat = new Quaternion(unityQuat.x, unityQuat.z, unityQuat.y, unityQuat.w);
        targetQuat[0] = swappedQuat.w;
        targetQuat[1] = swappedQuat.x;
        targetQuat[2] = swappedQuat.y;
        targetQuat[3] = swappedQuat.z;
    }
}
