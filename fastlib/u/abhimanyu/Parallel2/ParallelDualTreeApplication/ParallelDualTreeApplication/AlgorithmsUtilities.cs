using System;
using System.Collections.Generic;
using System.Text;
using Algorithms;
namespace Utilities
{
    public class NNResultsReader : SqlBulkCopyReader
    {
        private static String CREATE_RESULTS_TABLE =
            " Create Table <TableName> ( QueryId int, ReferenceId int, Distance float ) ";

        private List<DualTreeNN.OutputData> results;          // the nodes of the tree
        private int size;                   // number of nodes
        private int count;                  
        private const int fieldCount = 3;   // number of columns in tree table
        private DualTreeNN.OutputData current;

        public static String GetCreateResultsTableString(String rsltTableName)
        {
            String createStr = CREATE_RESULTS_TABLE;
            return createStr.Replace("<TableName>", rsltTableName);
        }

        public NNResultsReader(List<DualTreeNN.OutputData> rslt)
        {
            results = rslt;
            size = results.Count;
            count = -1;
        }

        override public bool Read()
        {
            count++;
            if (count < size)
            {
                current = results[count];
                return true;
            }
            return false;
        }

        override public object GetValue(int i)
        {
            switch (i)
            {
                case 0:
                    return current.queryPointId;
                case 1:
                    return current.referencePointId;
                case 2:
                    return current.distance;
            }
            return null;
        }

        override public int FieldCount
        {
            get { return fieldCount; }
        }
    }

}
