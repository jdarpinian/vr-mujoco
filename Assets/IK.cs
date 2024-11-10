using UnityEngine;
using Mujoco;
using System;
using System.Collections.Generic;

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
        }

        // Get the DOF indices for the actuators controlling the joints.
        _dofIndices = GetDofIndices(actuators);
    }

    void Update()
    {
        // Perform inverse kinematics to move the site towards the target.
        IKResult result = QPosFromSitePose();

        if (!result.success)
        {
            Debug.LogWarning($"IK did not converge: error norm = {result.errNorm}");
        }
    }

    public struct IKResult
    {
        public double[] qpos;
        public double errNorm;
        public int steps;
        public bool success;
    }

    IKResult QPosFromSitePose()
    {
        unsafe
        {
            // Initialize variables.
            bool success = false;
            int steps = 0;
            double errNorm = 0.0;

            int nv = _model->nv;
            int nq = _model->nq;

            // Target position and orientation.
            double[] targetPos = new double[3] { target.position.x, target.position.y, target.position.z };
            Quaternion unityQuat = target.rotation;
            double[] targetQuat = new double[4] { unityQuat.w, unityQuat.x, unityQuat.y, unityQuat.z };

            // Initialize arrays for the Jacobian and error.
            double[] jac = new double[6 * nv];
            double[] err = new double[6];
            double[] updateNV = new double[nv];

            // Main IK loop.
            for (steps = 0; steps < maxSteps; steps++)
            {
                errNorm = 0.0;

                // Perform forward kinematics.
                MujocoLib.mj_fwdPosition(_model, _data);

                // Get the current site position and orientation.
                double[] siteXPos = new double[3];
                double[] siteXMat = new double[9];

                for (int i = 0; i < 3; i++)
                {
                    siteXPos[i] = _data->site_xpos[_siteId * 3 + i];
                }
                for (int i = 0; i < 9; i++)
                {
                    siteXMat[i] = _data->site_xmat[_siteId * 9 + i];
                }

                // Compute positional error.
                double[] errPos = new double[3];
                for (int i = 0; i < 3; i++)
                {
                    errPos[i] = targetPos[i] - siteXPos[i];
                }
                errNorm += Norm(errPos);

                // Compute rotational error.
                double[] siteXQuat = new double[4];
                fixed (double* siteXQuatPtr = siteXQuat)
                fixed (double* siteXMatPtr = siteXMat)
                {
                    MujocoLib.mju_mat2Quat(siteXQuatPtr, siteXMatPtr);
                }
                double[] negSiteXQuat = new double[4];
                fixed (double* negSiteXQuatPtr = negSiteXQuat)
                fixed (double* siteXQuatPtr = siteXQuat)
                {
                    MujocoLib.mju_negQuat(negSiteXQuatPtr, siteXQuatPtr);
                }

                double[] errRotQuat = new double[4];
                fixed (double* errRotQuatPtr = errRotQuat)
                fixed (double* targetQuatPtr = targetQuat)
                fixed (double* negSiteXQuatPtr = negSiteXQuat)
                {
                    MujocoLib.mju_mulQuat(errRotQuatPtr, targetQuatPtr, negSiteXQuatPtr);
                }

                double[] errRot = new double[3];
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
                double[] jacp = new double[3 * nv];
                double[] jacr = new double[3 * nv];
                fixed (double* jacpPtr = jacp)
                fixed (double* jacrPtr = jacr)
                {
                    MujocoLib.mj_jacSite(_model, _data, jacpPtr, jacrPtr, _siteId);
                }

                // Combine position and rotation Jacobians.
                Array.Copy(jacp, 0, jac, 0, jacp.Length);
                Array.Copy(jacr, 0, jac, jacp.Length, jacr.Length);

                // Combine position and rotation errors.
                Array.Copy(errPos, 0, err, 0, errPos.Length);
                Array.Copy(errRot, 0, err, errPos.Length, errRot.Length);

                // Extract the Jacobian columns for the specified DOFs.
                double[,] jacJoints = ExtractJacobianColumns(jac, _dofIndices);

                // Solve for joint updates using the nullspace method.
                double[] updateJoints = NullspaceMethod(jacJoints, err, regularizationStrength);

                double updateNorm = Norm(updateJoints);

                // Check for sufficient progress.
                double progressCriterion = errNorm / updateNorm;
                if (progressCriterion > progressThreshold)
                {
                    Debug.LogWarning($"Insufficient progress at step {steps}: progress criterion = {progressCriterion}");
                    break;
                }

                // Limit the update norm.
                if (updateNorm > maxUpdateNorm)
                {
                    for (int i = 0; i < updateJoints.Length; i++)
                    {
                        updateJoints[i] *= maxUpdateNorm / updateNorm;
                    }
                }

                // Zero out the update vector.
                Array.Clear(updateNV, 0, updateNV.Length);

                // Assign updates to the appropriate DOF indices.
                for (int i = 0; i < _dofIndices.Length; i++)
                {
                    updateNV[_dofIndices[i]] = updateJoints[i];
                }

                // Integrate positions.
                unsafe
                {
                    fixed (double* updateNVPtr = updateNV)
                    {
                        MujocoLib.mj_integratePos(_model, _data->qpos, updateNVPtr, 1);
                    }
                }

                // Ensure the positions are updated in the simulation.
                MujocoLib.mj_fwdPosition(_model, _data);
            }

            // Prepare the result.
            double[] qpos = new double[nq];
            unsafe
            {
                for (int i = 0; i < nq; i++)
                {
                    qpos[i] = _data->qpos[i];
                }
            }

            return new IKResult
            {
                qpos = qpos,
                errNorm = errNorm,
                steps = steps,
                success = success
            };
        }
    }

    int[] GetDofIndices(MjActuator[] actuators)
    {
        unsafe
        {
            // Collect all DOF indices associated with the joints controlled by the actuators.
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
        // Determine the number of DOFs based on joint type.
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

    double[,] ExtractJacobianColumns(double[] jac, int[] dofIndices)
    {
        unsafe
        {
            int errDim = jac.Length / _model->nv;
            double[,] jacJoints = new double[errDim, dofIndices.Length];

            for (int i = 0; i < errDim; i++)
            {
                for (int j = 0; j < dofIndices.Length; j++)
                {
                    jacJoints[i, j] = jac[i * _model->nv + dofIndices[j]];
                }
            }

            return jacJoints;
        }
    }

    double[] NullspaceMethod(double[,] jacJoints, double[] delta, double regularizationStrength)
    {
        int n = jacJoints.GetLength(1); // Number of joints
        int m = jacJoints.GetLength(0); // Error dimension

        double[,] hessApprox = new double[n, n];
        double[] jointDelta = new double[n];

        // Compute J^T * J
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double sum = 0.0;
                for (int k = 0; k < m; k++)
                {
                    sum += jacJoints[k, i] * jacJoints[k, j];
                }
                hessApprox[i, j] = sum;
            }
        }

        // Add regularization
        for (int i = 0; i < n; i++)
        {
            hessApprox[i, i] += regularizationStrength;
        }

        // Compute J^T * delta
        for (int i = 0; i < n; i++)
        {
            double sum = 0.0;
            for (int k = 0; k < m; k++)
            {
                sum += jacJoints[k, i] * delta[k];
            }
            jointDelta[i] = sum;
        }

        // Solve the linear system
        double[] dq = SolveLinearSystem(hessApprox, jointDelta);
        return dq;
    }

    double[] SolveLinearSystem(double[,] A, double[] b)
    {
        // Use a simple linear solver (e.g., Gaussian elimination)
        // For brevity, we'll use a placeholder method here.
        // In practice, you should use a reliable linear algebra library.
        int n = b.Length;
        double[] x = new double[n];

        // Placeholder implementation (this needs to be replaced with a real solver)
        // You can use libraries like Math.NET Numerics for robust solutions.
        for (int i = 0; i < n; i++)
        {
            x[i] = b[i] / (A[i, i] + 1e-6); // Avoid division by zero
        }

        return x;
    }

    double Norm(double[] vec)
    {
        double sum = 0.0;
        foreach (var v in vec)
        {
            sum += v * v;
        }
        return Math.Sqrt(sum);
    }
}