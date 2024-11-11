using UnityEngine;
using Mujoco;
using System;
using System.Collections.Generic;
using Accord.Math;
using Accord.Math.Optimization;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System.Linq;

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
    private double[] qpos;

    // Variables for the IK loop.
    private int nv;
    private int nq;

    // Joint limits per qpos index.
    private double[] lowerLimits;
    private double[] upperLimits;
    private int[] qposIndices;

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
        deltaVector = Vector<double>.Build.Dense(errDim);

        // Collect joint limits from actuators.
        lowerLimits = new double[numJoints];
        upperLimits = new double[numJoints];

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

                int jointId = joint.MujocoId;
                int dofAdr = _model->jnt_dofadr[jointId];
                int dofNum = GetJointDofNum(_model->jnt_type[jointId]);

                Vector2 ctrlRange = actuator.CommonParams.CtrlRange;
                double lowerLimit = ctrlRange.x;
                double upperLimit = ctrlRange.y;

                for (int j = 0; j < dofNum; j++)
                {
                    int index = i * dofNum + j;
                    lowerLimits[index] = lowerLimit;
                    upperLimits[index] = upperLimit;
                }
            }
        }
    }

    unsafe void Update()
    {
        // Create a copy of mjData
        MujocoLib.mjData_* dataCopy = MujocoLib.mj_makeData(_model);

        try
        {
            // Copy the current data into dataCopy
            MujocoLib.mj_copyData(dataCopy, _model, _data);

            // Perform inverse kinematics on the copied data
            IKResult result = QPosFromSitePose(_model, dataCopy);

            if (!result.success)
            {
                Debug.LogWarning($"IK did not converge: error norm = {result.errNorm}");
            }

            // Use the actuators to drive the joints toward the IK result
            SetActuatorsToIKResult(result.qpos);
        }
        finally
        {
            // Free the copied data to prevent memory leaks
            MujocoLib.mj_deleteData(dataCopy);
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
        var targetPosition = target.position;
        targetPos[0] = targetPosition.x;
        targetPos[1] = targetPosition.z;
        targetPos[2] = targetPosition.y;

        Quaternion unityQuat = target.rotation;
        ConvertUnityQuatToMujoco(unityQuat, targetQuat);

        // Get current qpos
        double[] qposCurrent = new double[_dofIndices.Length];
        for (int i = 0; i < _dofIndices.Length; i++)
        {
            qposCurrent[i] = data->qpos[_dofIndices[i]];
        }

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

            // Solve for joint updates using the QP solver with joint limits
            double[] dq = SolveQP(jacJoints, deltaVector.ToArray(), regularizationStrength, qposCurrent, lowerLimits, upperLimits);

            double updateNorm = Norm(dq);

            if (updateNorm > maxUpdateNorm)
            {
                double scale = maxUpdateNorm / updateNorm;
                for (int i = 0; i < dq.Length; i++)
                {
                    dq[i] *= scale;
                }
            }

            // Update qposCurrent
            for (int i = 0; i < qposCurrent.Length; i++)
            {
                qposCurrent[i] += dq[i];
                // Ensure qposCurrent is within limits
                qposCurrent[i] = Math.Max(lowerLimits[i], Math.Min(upperLimits[i], qposCurrent[i]));
                // Update data->qpos
                data->qpos[_dofIndices[i]] = qposCurrent[i];
            }

            // Perform forward kinematics with updated positions
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

    unsafe void SetActuatorsToIKResult(double[] targetQpos)
    {
        // For each actuator, set its Control property to the desired joint position
        for (int i = 0; i < actuators.Length; i++)
        {
            var actuator = actuators[i];
            var joint = actuator.Joint;
            if (joint == null)
            {
                continue;
            }

            int jointId = joint.MujocoId;
            int qposAdr = _model->jnt_qposadr[jointId];

            // Get the desired position for this joint
            double desiredQpos = targetQpos[qposAdr];

            // Set the actuator's Control property and _data->ctrl
            actuator.Control = (float)desiredQpos;
            unsafe
            {
                _data->ctrl[actuator.MujocoId] = desiredQpos;
            }
        }
    }

    unsafe int[] GetDofIndices(MjActuator[] actuators)
    {
        List<int> dofIndices = new List<int>();

        foreach (var actuator in actuators)
        {
            var joint = actuator.Joint;
            if (joint == null)
            {
                continue;
            }

            int jointId = joint.MujocoId;
            int dofAdr = _model->jnt_dofadr[jointId];
            int dofNum = GetJointDofNum(_model->jnt_type[jointId]);

            for (int i = 0; i < dofNum; i++)
            {
                dofIndices.Add(dofAdr + i);
            }
        }

        return dofIndices.ToArray();
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
        // Swap Y and Z axes to convert from Unity to MuJoCo coordinate system
        Quaternion swappedQuat = new Quaternion(unityQuat.x, unityQuat.z, unityQuat.y, unityQuat.w);

        // Rearrange components to MuJoCo's (w, x, y, z) format
        targetQuat[0] = swappedQuat.w;
        targetQuat[1] = swappedQuat.x;
        targetQuat[2] = swappedQuat.y;
        targetQuat[3] = swappedQuat.z;
    }

    double[] SolveQP(Matrix<double> jacJoints, double[] delta, double regularizationStrength,
                     double[] qposCurrent, double[] lowerLimits, double[] upperLimits)
    {
        int n = jacJoints.ColumnCount; // Number of joints

        // Formulate the QP problem: minimize 0.5 * x^T H x - f^T x
        // H = J^T J + regularization * I
        // f = -J^T delta

        Matrix<double> H = jacJoints.TransposeThisAndMultiply(jacJoints);
        if (regularizationStrength > 0)
        {
            H = H + Matrix<double>.Build.DenseIdentity(n) * regularizationStrength;
        }

        Vector<double> f = -jacJoints.TransposeThisAndMultiply(Vector<double>.Build.Dense(delta));

        // Convert H and f to Accord.NET formats
        double[,] HAccord = H.ToArray();
        double[] fAccord = f.ToArray();

        // Define the objective function
        var objectiveFunction = new QuadraticObjectiveFunction(HAccord, fAccord);

        // Define inequality constraints for joint limits
        List<LinearConstraint> constraints = new List<LinearConstraint>();
        for (int i = 0; i < n; i++)
        {
            // Create CombinedAs array with zeros and a single 1.0 at index i
            double[] combinedUpper = new double[n];
            double[] combinedLower = new double[n];
            combinedUpper[i] = 1.0;
            combinedLower[i] = 1.0;

            // Upper limit constraint: dq[i] <= upperLimits[i] - qposCurrent[i]
            constraints.Add(new LinearConstraint(numberOfVariables: n)
            {
                VariablesAtIndices = Enumerable.Range(0, n).ToArray(),
                CombinedAs = combinedUpper,
                ShouldBe = ConstraintType.LesserThanOrEqualTo,
                Value = upperLimits[i] - qposCurrent[i]
            });

            // Lower limit constraint: dq[i] >= lowerLimits[i] - qposCurrent[i]
            constraints.Add(new LinearConstraint(numberOfVariables: n)
            {
                VariablesAtIndices = Enumerable.Range(0, n).ToArray(),
                CombinedAs = combinedLower,
                ShouldBe = ConstraintType.GreaterThanOrEqualTo,
                Value = lowerLimits[i] - qposCurrent[i]
            });
        }

        // Create and solve the QP problem
        var solver = new GoldfarbIdnani(objectiveFunction, constraints);

        bool success = solver.Minimize();
        if (!success)
        {
            Debug.LogError("QP Solver failed to find a solution.");
            return new double[n];
        }

        return solver.Solution;
    }

}