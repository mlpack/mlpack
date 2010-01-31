using System;
using System.Diagnostics;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.Serialization;

namespace KDTreeStructures
{
    // considering splitting this clas into 2
    [Serializable()]
    public sealed class KNode :  ISerializable
    {
        // these are set in constructor
        public double splitPoint;  // -1 if leaf
        public int splitDimension; // -1 if leaf
        private int nodeId;
        // these are set using setters
        private int lcIndex, rcIndex;   // -1 if leaf

        // mr data
        public int numPoints;
        public int level;
        private int boxIdx;
        private KDTree tree;
        
        public KNode(SerializationInfo info, StreamingContext ctxt)
        {
            //Get the values from info and assign them to the appropriate properties
            splitPoint = (int)info.GetValue("splitPoint", typeof(double));
            splitDimension = (int)info.GetValue("splitDimension", typeof(int));
            nodeId = (int)info.GetValue("nodeId", typeof(int));
            lcIndex = (int)info.GetValue("lcIndex", typeof(int));
            rcIndex = (int)info.GetValue("rcIndex", typeof(int));
            numPoints = (int)info.GetValue("numPoints", typeof(int));
            level = (int)info.GetValue("level", typeof(int));
            boxIdx = (int)info.GetValue("boxIdx", typeof(int));

        }
        
        //Serialization function.
        public void GetObjectData(SerializationInfo info, StreamingContext ctxt)
        {
            info.AddValue("splitPoint", splitPoint);
            info.AddValue("splitDimension", splitDimension );
            info.AddValue("nodeId", nodeId);
            info.AddValue("lcIndex", lcIndex);
            info.AddValue("rcIndex", rcIndex);
            info.AddValue("numPoints", numPoints);
            info.AddValue("level", level);
            info.AddValue("boxIdx", boxIdx);
        }

        public KNode(int splitDim, double splitPt, int id,
            int boxIndex, int numPts, int lvl, KDTree ownerTree)
        {
            
            splitDimension = splitDim;
            splitPoint = splitPt;
            nodeId = id;
            boxIdx = boxIndex;
            tree = ownerTree;
            numPoints = numPts;
            level = lvl;
            lcIndex = -1;
            rcIndex = -1;
        }
       

        public int GetBoxIdx()
        {
            return boxIdx;
        }
        public KDTree GetTree()
        {
            return tree;
        }


        public bool IsLeaf()
        {
            return lcIndex == -1 ? true : false;
        }

       
        

        public KNode GetChild(int i)
        {
            if (i == 0)
            {
                return tree.GetNode(lcIndex);
            }
            else if (i == 1)
            {
                return tree.GetNode(rcIndex);
            }
            else
            {
                throw new Exception();
            }

        }

        public int GetLC()
        {
            return lcIndex;
        }

        public int GetRC()
        {
            return rcIndex;
        }

        public void SetLC(int index)
        {
            lcIndex = index;
        }

        public void SetRC(int index)
        {
            rcIndex = index;
        }

        public int GetNumPoints()
        {
            return numPoints;
        }

        public int GetLevel()
        {
            return level;
        }

        public int GetNodeId()
        {
            return nodeId;
        }

        public double GetSplitPoint()
        {
            return splitPoint;
        }

        public int GetSplitDimension()
        {
            return splitDimension;
        }

        public void SetNodeId( int id )
        {
            nodeId = id;
        }

        public void SetBoxIdx(int idx)
        {
            boxIdx = idx;
        }

       
    }

}