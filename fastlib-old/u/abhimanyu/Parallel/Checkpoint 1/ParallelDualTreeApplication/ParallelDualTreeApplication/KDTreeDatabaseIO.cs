using System;
using System.Text;
using Utilities;
using System.Collections.Generic;
using System.Data.SqlClient;

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
            if (count < size)
            {
                current = nodes[count];
                return true;
            }
            return false;
        }

        private String GetMinBoxString()
        {
            StringBuilder str = new StringBuilder("");
            for (int i = 0; i < k - 1; ++i)
            {
                str.Append(mins[k * count + i]);
                str.Append(",");
            }
            str.Append(mins[k * count + k - 1]);
            return str.ToString();
        }

        private String GetMaxBoxString()
        {
            StringBuilder str = new StringBuilder("");
            for (int i = 0; i < k - 1; ++i)
            {
                str.Append(maxs[k * count + i]);
                str.Append(",");
            }
            str.Append(maxs[k * count + k - 1]);
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
            }
            return null;
        }
        override public int FieldCount
        {
            get { return fieldCount; }
        }

        /// <summary>
        /// Since this class is used by the KDTree to write its node data to the data base
        /// this function mirrors the write functionality to do the opposite, initialize the
        /// tree from a table previously stored by this class.
        /// </summary>
        /// <param name="dbName"></param>
        /// <param name="tableName"></param>
        /// <param name="boxMins"></param>
        /// <param name="boxMaxs"></param>
        /// <param name="nodeList"></param>
        public static void ReadNodes(String dbName, String tableName,
            double[] boxMins, double[] boxMaxs, List<KNode> nodeList, int k, KDTree tree)
        {
            SqlConnection connection = new SqlConnection();
            //connection.ConnectionString = "Context Connection=true";
            connection.ConnectionString = "Data Source=KLEENE; Initial Catalog=Master; Integrated Security=True;";
            connection.Open();
            String queryText = "Select * from " + dbName + ".dbo." + tableName + " order by NodeId ";
            SqlCommand comm = new SqlCommand(queryText, connection);
            SqlDataReader rdr = comm.ExecuteReader();

            int boxIndex = 0;

            unsafe
            {
                fixed (double* minPtr = boxMins, maxPtr = boxMaxs)
                {
                    double* mn = minPtr, mx = maxPtr;
                    while (rdr.Read())
                    {
                        KNode node = new KNode(rdr.GetInt32(1), rdr.GetDouble(2), rdr.GetInt32(0),
                            boxIndex, rdr.GetInt32(4), rdr.GetInt32(3), tree);
                        node.SetLC(rdr.GetInt32(5));
                        node.SetRC(rdr.GetInt32(6));

                        String minStr = rdr.GetString(7);
                        String maxStr = rdr.GetString(8);
                        String[] minDoubles = minStr.Split(',');
                        String[] maxDoubles = maxStr.Split(',');

                        for (int i = 0; i < k; ++i)
                        {
                            *mn++ = Double.Parse(minDoubles[i]);
                            *mx++ = Double.Parse(maxDoubles[i]);
                        }

                        boxIndex++;
                        nodeList.Add(node);

                    }
                }
            }
        }

    }


}
