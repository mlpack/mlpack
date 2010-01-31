using System;
using System.Text;
using Utilities;
using System.Collections.Generic;

namespace KDTreeStructures
{
    class KDTreeDataReader : SqlBulkCopyReader
    {
        private List<KNode> nodes;
        double[] mins, maxs;
        private int k;
        private int size;
        
        private int count;
        private const int fieldCount = 9;
        private KNode current;
        
        
        public KDTreeDataReader(List<KNode> nodeList, 
            double[] boxMins, double[] boxMaxs, int k, int dataSize)
        {
            nodes = nodeList;
            mins = boxMins;
            maxs = boxMaxs;
            this.k = k;
            size = dataSize;
            count = -1;
        }


        override public bool Read()
        {
            count++;
            if( count < size )
            {
                current = nodes[count];
                return true;
            }
            return false;
        }

        private String GetMinBoxString()
        {
            StringBuilder str = new StringBuilder("[");
            for (int i = 0; i < k; ++i)
            {
                str.Append(mins[k * count + i]);
                str.Append(",");
            }
            str.Append("]");
            return str.ToString();
        }

        private String GetMaxBoxString()
        {
            StringBuilder str = new StringBuilder("[");
            for (int i = 0; i < k; ++i)
            {
                str.Append(mins[k * count + i]);
                str.Append(",");
            }
            str.Append("]");
            return str.ToString();
        }
        override public object GetValue(int i)
        {
            switch (i)
            {
                case 0:
                    return current.GetNodeId();
                case 1:
                    return current.GetSplitDimension();
                case 2:
                    return current.GetSplitPoint();
                case 3:
                    return current.GetLevel();
                case 4:
                    return current.GetNumPoints();
                case 5:
                    return current.GetLC();
                case 6:
                    return current.GetRC();
                case 7:
                    return GetMinBoxString();
                case 8:
                    return GetMaxBoxString();
                default:
                    throw new Exception();
            }
        }
        override public int FieldCount 
        { 
            get { return fieldCount; }
        }

    }
    
    class KDTreeDatabaseIO
    {

    }
}
