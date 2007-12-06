using System;
using System.Collections.Generic;
using System.Text;
using System.Diagnostics;
using KDTreeStructures;
using Utilities;
using Algorithms;

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

        static void MainParallel(string[] args)
        {
                
                DualTreeNN nn = new DualTreeNN();
                
                nn.GetNearestNeighbors("aerial5000Tree", 50);

                nn.SaveResultsToDb("nntest", "aerial5000Results_pdt");
        }

        static void Main(string[] args)
        {

            using (new MPI.Environment(ref args))
            {
                int rank = MPI.Communicator.world.Rank;
                if (rank == 0)
                {
                    SetupEnvironment();
                }
                //MainCreateTree(args);
                MainParallel(args);

                if (rank == 0)
                    FinalizeEnvironment();
            }
        }

        static void MainCreateTree(string[] args)
        {

            int rank = MPI.Communicator.world.Rank;
            int[] ids;
            double[] data;
            DatabaseUtilities.GetTableData("nntest", "aerial5000", out ids, out data);
            // pseudo parallel runtimes
            MPI.Communicator.world.Barrier();
            if (rank != 0)
            {
                MPI.Communicator.world.Receive<int>(rank - 1, 0);
            }

            KDTree tree = null;
            double start = MPI.Environment.Wtime;
            for (int i = 0; i < 10; i++)
            {
                tree = new KDTree();
                tree.InitializeTree(ids, data);
            }
            double end = MPI.Environment.Wtime;
            double t1 = end - start;
            if (rank != MPI.Communicator.world.Size - 1)
            {
                MPI.Communicator.world.Send<int>(0, rank + 1, 0);
            }

            MPI.Communicator.world.Barrier();
            double t2 = 0;
            for (int i = 0; i < 10; i++)
            {
                tree = new KDTree();
                tree.InitializeTree(ids, data);
                MPI.Communicator.world.Barrier();
                start = MPI.Environment.Wtime;
                tree.CollateSubtrees();
                end = MPI.Environment.Wtime;
                t2 += end - start;
            }
            MPI.Communicator.world.Barrier();
            Console.WriteLine("Rank " + MPI.Communicator.world.Rank
                + "Time taken per iteration = " + (t1 / 10));
            Console.WriteLine("Rank " + MPI.Communicator.world.Rank
                + "Time taken for commuinication = " + (t2 / 10));
            //tree.SaveToDb("nntest", "aerialLarge1m", "nntest", "pAerialLarge1mTree");
            
            

        }
    }
}
