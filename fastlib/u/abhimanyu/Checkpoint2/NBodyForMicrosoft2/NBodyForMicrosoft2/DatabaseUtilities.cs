using System;
using System.Diagnostics;
using System.Data.SqlClient;



namespace Utilities
{
    /// <summary>
    /// This class provides all the database utilities needed by the system.
    /// Funtionality includes returning the data from tables, returning 
    /// table sizes, dimensions etc.
    /// </summary>
    public sealed class DatabaseUtilities
    {
        // DEBUGGING ELEMENTS
        private static TextWriterTraceListener Tracer = 
            new TextWriterTraceListener("C:\\Documents and Settings\\manyu\\Desktop\\trace.txt");
        static DateTime startTime = DateTime.Now;
        static DateTime endTime = DateTime.Now;

        /// <summary>
        /// This function returns the data from the table in row major order.
        /// It also returns the row count and dimensionality. Note the first column is 
        /// currently assumed to be the id column and this information is not returned
        /// 
        /// </summary>
        /// <param name="dbName"></param>
        /// <param name="tableName"></param>
        /// <param name="rowCount"></param>
        /// <returns></returns>
        public static void GetTableData(String dbName, String tableName,
            out int[] ids, out double[] data )
        {
            /**
             * open connection to database
             */
            Trace.Listeners.Add(Tracer);
            Trace.WriteLine("DatabaseUtilities.GetTableData()");
            SqlConnection connection = new SqlConnection();
            //connection.ConnectionString = "Context Connection=true";
            connection.ConnectionString = "Data Source=KLEENE; Initial Catalog=NNTest; Integrated Security=True;";
            startTime = DateTime.Now;
            connection.Open();
            endTime = DateTime.Now;
            Trace.WriteLine("DatabaseUtilities.GetTableData()- Getting DB Connection took " + PrintTime());

            /**
             * get the number of rows in the table
             */
            startTime = DateTime.Now;
            int rowCount = GetNumberOfRows(connection, dbName, tableName);
            endTime = DateTime.Now;
            Trace.WriteLine("DatabaseUtilities.GetTableData()- Getting row count took " + PrintTime());
            
            /**
             * Execute select query
             */ 
            SqlCommand cmd = new SqlCommand();
            cmd.CommandText = "SELECT * FROM " + dbName + ".dbo." + tableName;
            cmd.CommandType = System.Data.CommandType.Text;
            cmd.Connection = connection;
            startTime = DateTime.Now;
            SqlDataReader rdr = cmd.ExecuteReader();
            endTime = DateTime.Now;
            Trace.WriteLine("DatabaseUtilities.GetTableData()- Executing Select Statement took" + PrintTime());;
            
            /**
             * ITERATE
             */ 
            int N = rdr.VisibleFieldCount - 1;    // assuming that 1 column is the id
            ids = new int[rowCount];
            data = new double[rowCount * N];
            object[] rowData = new object[N+1];
            startTime = DateTime.Now;
            unsafe
            {
                fixed (double* dataPtr = data)
                {
                    fixed(int* idPtr = ids)
                    {
                        double *dt = dataPtr;
                        int* id = idPtr;
                        while (rdr.Read())
                        {
                            rdr.GetValues(rowData);
                            *id = (int)rowData[0];
                            ++id;
                            for (int i = 0; i < N; ++i)
                            {
                                *dt = (double)rowData[i + 1];
                                ++dt;
                            }
                        }
                    }
                }
            }
            endTime = DateTime.Now;
            Trace.WriteLine("DatabaseUtilities.GetTableData()- Iterating took " + PrintTime());

            /**
             * Release resources.
             */
            rdr.Close();
            connection.Close();
        }

        /// <summary>
        /// Returns the number of rows in the table.
        /// </summary>
        /// <param name="connection"></param>
        /// <param name="dbName"></param>
        /// <param name="tableName"></param>
        /// <returns></returns>
        private static int GetNumberOfRows(SqlConnection connection, String dbName, String tableName)
        {
            SqlCommand cmd = new SqlCommand();
            cmd.Connection = connection;
            // get the number of rows in the table
            cmd.CommandText = "SELECT count(*) FROM " + dbName + ".dbo."+ tableName;
            cmd.CommandType = System.Data.CommandType.Text;
            SqlDataReader rdr = cmd.ExecuteReader();
            rdr.Read();
            int rowCount = rdr.GetInt32(0);
            rdr.Close();
            return rowCount;
        }

        private static String PrintTime()
        {
            TimeSpan duration = endTime - startTime;
            /*
            return
                duration.Hours
                + ":" + duration.Minutes
                + ":" + duration.Seconds + "." + duration.Milliseconds;
             */
            return duration.Ticks.ToString();
        }
    }
}
