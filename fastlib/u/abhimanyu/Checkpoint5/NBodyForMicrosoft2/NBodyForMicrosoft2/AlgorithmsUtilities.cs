using System;
using System.Collections.Generic;
using System.Text;
using Algorithms;

namespace Utilities
{
    /// <summary>
    /// This class is the Data Reader used by SqlBulkCopy to copy data into
    /// the Nearest Neighbor's results table in the database.
    /// </summary>
    public class NNResultsReader : SqlBulkCopyReader
    {
        // The create table query for the NN results
        private static String CREATE_RESULTS_TABLE =
            " Create Table <TableName> ( QueryId int, ReferenceId int, Distance float ) ";

        private List<DualTreeNN.OutputData> results;          // the nodes of the tree
        private int size;                   // number of nodes
        private int count;                  
        private const int fieldCount = 3;   // number of columns in tree table
        private DualTreeNN.OutputData current;

        /// <summary>
        /// Returns the query string to be used to create the results table.
        /// </summary>
        /// <param name="rsltTableName">The name of the results table.</param>
        public static String GetCreateResultsTableString(String rsltTableName)
        {
            String createStr = CREATE_RESULTS_TABLE;
            return createStr.Replace("<TableName>", rsltTableName);
        }

        /// <summary>
        /// The results to be read into a Database Table.
        /// </summary>
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
