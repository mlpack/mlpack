using System;
using System.Diagnostics;
using System.Data.SqlClient;
using System.Data;



namespace Utilities
{
    /// <summary>
    /// Bulk Reader to help write large amounts of data to the 
    /// sql server fast.
    /// </summary>
    public abstract class SqlBulkCopyReader : IDataReader
    {
        // derived must implement only these three
        public abstract bool Read();
        public abstract object GetValue(int i);
        public abstract int FieldCount { get; }

        // empty methods derived classes may want to implement
        public virtual void Close() { }
        public virtual void Dispose() { }
        public virtual int GetOrdinal(string name) { throw new NotImplementedException(); }
        public virtual object this[int i] { get { throw new NotImplementedException(); } }
        public virtual int Depth { get { throw new NotImplementedException(); } }
        public virtual bool IsClosed { get { throw new NotImplementedException(); } }
        public virtual int RecordsAffected { get { throw new NotImplementedException(); } }
        public virtual DataTable GetSchemaTable() { throw new NotImplementedException(); }
        public virtual bool NextResult() { throw new NotImplementedException(); }
        public virtual object this[string name] { get { throw new NotImplementedException(); } }
        public virtual bool GetBoolean(int i) { throw new NotImplementedException(); }
        public virtual byte GetByte(int i) { throw new NotImplementedException(); }
        public virtual long GetBytes(int i, long fieldOffset, byte[] buffer, int bufferoffset, int length) { throw new NotImplementedException(); }
        public virtual char GetChar(int i) { throw new NotImplementedException(); }
        public virtual long GetChars(int i, long fieldoffset, char[] buffer, int bufferoffset, int length) { throw new NotImplementedException(); }
        public virtual IDataReader GetData(int i) { throw new NotImplementedException(); }
        public virtual string GetDataTypeName(int i) { throw new NotImplementedException(); }
        public virtual DateTime GetDateTime(int i) { throw new NotImplementedException(); }
        public virtual decimal GetDecimal(int i) { throw new NotImplementedException(); }
        public virtual double GetDouble(int i) { throw new NotImplementedException(); }
        public virtual Type GetFieldType(int i) { throw new NotImplementedException(); }
        public virtual float GetFloat(int i) { throw new NotImplementedException(); }
        public virtual Guid GetGuid(int i) { throw new NotImplementedException(); }
        public virtual short GetInt16(int i) { throw new NotImplementedException(); }
        public virtual int GetInt32(int i) { throw new NotImplementedException(); }
        public virtual long GetInt64(int i) { throw new NotImplementedException(); }
        public virtual string GetName(int i) { throw new NotImplementedException(); }
        public virtual string GetString(int i) { throw new NotImplementedException(); }
        public virtual int GetValues(object[] values) { throw new NotImplementedException(); }
        public virtual bool IsDBNull(int i) { throw new NotImplementedException(); }
    }

    /// <summary>
    /// This class provides all the general database utilities needed by the system.
    /// Funtionality includes returning the data from tables, returning 
    /// table sizes, dimensions etc.
    /// </summary>
    public sealed class DatabaseUtilities
    {
        public static int TIMEOUT = 600;
        
        // DEBUGGING ELEMENTS
        private static TextWriterTraceListener Tracer = 
            new TextWriterTraceListener("C:\\Documents and Settings\\manyu\\Desktop\\trace.txt");
        static DateTime startTime = DateTime.Now;
        static DateTime endTime = DateTime.Now;

        

        /// <summary>
        /// This function returns the data from the table in row major order.
        /// It also returns the row count and dimensionality. Note the first column is 
        /// currently assumed to be the id column and this information is not returned
        /// </summary>
        public static void GetTableData(String dbName, String tableName,
            out int[] ids, out double[] data )
        {
            /**
             * open connection to database
             */
            Trace.Listeners.Add(Tracer);
            //Trace.WriteLine("DatabaseUtilities.GetTableData()");
            SqlConnection connection = new SqlConnection();
            //connection.ConnectionString = "Context Connection=true";
            connection.ConnectionString = "Data Source=KLEENE; Initial Catalog=" + dbName + "; Integrated Security=True;";
            //startTime = DateTime.Now;
            connection.Open();
            //endTime = DateTime.Now;
            //Trace.WriteLine("DatabaseUtilities.GetTableData()- Getting DB Connection took " + PrintTime());

            /**
             * get the number of rows in the table
             */
            //startTime = DateTime.Now;
            int rowCount = GetNumberOfRows(connection, dbName, tableName);
            //endTime = DateTime.Now;
            Trace.WriteLine("DatabaseUtilities.GetTableData()- Getting row count " + rowCount);
            
            /**
             * Execute select query
             */ 
            SqlCommand cmd = new SqlCommand();
            cmd.CommandText = "SELECT * FROM " + dbName + ".dbo." + tableName;
            cmd.CommandType = System.Data.CommandType.Text;
            cmd.Connection = connection;
            //startTime = DateTime.Now;
            SqlDataReader rdr = cmd.ExecuteReader();
            //endTime = DateTime.Now;
            //Trace.WriteLine("DatabaseUtilities.GetTableData()- Executing Select Statement took" + PrintTime());;
            
            /**
             * ITERATE
             */ 
            int N = rdr.VisibleFieldCount - 1;    // assuming that 1 column is the id
            ids = new int[rowCount];
            data = new double[rowCount * N];
            object[] rowData = new object[N+1];
            //startTime = DateTime.Now;
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
            //endTime = DateTime.Now;
            //Trace.WriteLine("DatabaseUtilities.GetTableData()- Iterating took " + PrintTime());

            /**
             * Release resources.
             */
            rdr.Close();
            connection.Close();
        }

        /// <summary>
        /// Returns the number of rows in the table.
        /// </summary>
        private static int GetNumberOfRows(SqlConnection connection, String dbName, String tableName)
        {
            SqlCommand cmd = new SqlCommand();
            cmd.Connection = connection;
            // get the number of rows in the table
            cmd.CommandText = "SELECT count(*) FROM " + dbName + ".dbo."+ tableName;
            cmd.CommandType = System.Data.CommandType.Text;
            cmd.CommandTimeout = TIMEOUT;
            SqlDataReader rdr = cmd.ExecuteReader();
            rdr.Read();
            int rowCount = rdr.GetInt32(0);
            rdr.Close();
            return rowCount;
        }

        /*
        /// <summary>
        /// Prints the number of clock ticks elapsed between endTime and startTime.
        /// </summary>
        /// <returns></returns>
        private static String PrintTime()
        {
            TimeSpan duration = endTime - startTime;
            return duration.Ticks.ToString();
        }
        */
    }
}
