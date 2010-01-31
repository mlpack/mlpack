using System;
using System.Collections.Generic;
using System.Text;
using System.Diagnostics;
using KDTreeStructures;
using Utilities;

namespace ParallelDualTreeApplication
{
    class Program
    {
        private static TextWriterTraceListener Tracer;

        private static void SetupEnvironment()
        {
            Tracer = new
               TextWriterTraceListener("C:\\Documents and Settings\\manyu\\Desktop\\trace_mpi.txt");
            Trace.Listeners.Add(Tracer);
        }

        private static void FinalizeEnvironment()
        {
            Trace.Close();
        }

        static void Main(string[] args)
        {
            //try
           // {
                
                
                using (new MPI.Environment(ref args))
                {
                    int rank = MPI.Communicator.world.Rank;
                    if( rank == 0 )
                        SetupEnvironment();
                    Trace.WriteLine("...Rank " + rank + " here. checking in.");
                    KDTree tree = new KDTree();
                    int[] ids;
                    double[] data;
                    DatabaseUtilities.GetTableData("nntest", "aeriallarge1m", out ids, out data);
                    tree.MPIInitializeTree(ids, data);
                    tree.SaveToDb("nntest", "aerialLarge1m", "nntest", "pAerialLarge1mTree");
                    Trace.WriteLine("...Rank " + MPI.Communicator.world.Rank + " here. checking out.");
                    if( rank == 0 )
                        FinalizeEnvironment();
                }
                
                
            //}
            //catch (Exception e)
           // {
             //   FinalizeEnvironment();
              //  throw e;
            //}
            
        }
    }
}
