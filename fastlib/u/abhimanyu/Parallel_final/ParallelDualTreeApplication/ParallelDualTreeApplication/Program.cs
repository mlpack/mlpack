using System;
using System.Collections.Generic;
using System.Text;
using System.Diagnostics;
using KDTreeStructures;
using Utilities;
using Algorithms;
using System.IO;


namespace ParallelDualTreeApplication
{
    class Program
    {
        private static Dictionary<String,String> properties;
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

       

        private static void PrintUsage()
        {
            Console.WriteLine("Please enter the config file name");
        }
        static void Main1(string[] args)
        {
            using (new MPI.Environment(ref args))
            {
                int rank = MPI.Communicator.world.Rank;
               
                if (rank == 0)
                {
                    SetupEnvironment();
                }

                DualTreeNN nn = new DualTreeNN();
                nn.GetNearestNeighbors("Aerial400ktree", 4);
                nn.SaveResultsToDb("nntest", "temp_pdt");
                if (rank == 0)
                {
                    FinalizeEnvironment();
                }
            }
        }

        static void Main(string[] args)
        {
            using (new MPI.Environment(ref args))
            {
                int rank = MPI.Communicator.world.Rank;
                if (args.Length != 1)
                {
                    
                    if( rank == 0)
                    {
                        PrintUsage();
                    }
                    return;
                }
                ReadPropertiesFile(args[0]);
                if (rank == 0)
                {
                    SetupEnvironment();
                }

                String function;
                if (properties.TryGetValue("function", out function))
                {
                    if (function.Equals("tree"))
                    {
                        Console.WriteLine("here");
                        RunCreateTree();
                    }
                    else if (function.Equals("nn"))
                    {
                        RunNN();
                    }
                    else
                    {
                        throw new Exception("Cannot recognize function: " + function);
                    }

                }
                if (rank == 0)
                {
                    FinalizeEnvironment();
                }
            }
        }

        private static void RunNN()
        {
            String io;
            properties.TryGetValue("io", out io);
            if (io.Equals("file"))
            {
                String treeFile, mappingFile, nnFile, kNN;
                properties.TryGetValue("treeFile", out treeFile);
                properties.TryGetValue("mappingFile", out mappingFile);
                properties.TryGetValue("knn", out kNN);
                properties.TryGetValue("nnFile", out nnFile);
                DualTreeNN nn = new DualTreeNN();
                nn.GetNearestNeighbors(treeFile, mappingFile, int.Parse(kNN));
                
                bool append = false;
                if (MPI.Communicator.world.Rank != 0)
                {
                    MPI.Communicator.world.Receive<int>(MPI.Communicator.world.Rank - 1, 0);
                    append = true;
                }
                nn.SaveResultsToFile(nnFile, append);
                if (MPI.Communicator.world.Rank != MPI.Communicator.world.Size - 1)
                {
                    MPI.Communicator.world.Send<int>(0, MPI.Communicator.world.Rank + 1, 0);
                }


            }
            else if (io.Equals("db"))
            {
                String treeTable, kNN, nnDb, nnTable;
                properties.TryGetValue("treeTable", out treeTable);
                properties.TryGetValue("knn", out kNN);
                properties.TryGetValue("nnDb", out nnDb);
                properties.TryGetValue("nnTable", out nnTable);
                DualTreeNN nn = new DualTreeNN();
                nn.GetNearestNeighbors(treeTable, int.Parse(kNN));
                nn.SaveResultsToDb(nnDb, nnTable);
            }
            else
            {
                throw new Exception();
            }
        }

        private static void RunCreateTree()
        {
            
            String io;
            properties.TryGetValue("io", out io);
            KDTree tree = new KDTree();
            int[] ids;
            double[] data;
            if (io.Equals("file"))
            {
                
                // input parameters
                String dataFile, colSeperator, ignoreFirst;
                properties.TryGetValue("dataFile", out dataFile);
                properties.TryGetValue("colSeperator", out colSeperator);
                properties.TryGetValue("ignoreFirstLine", out ignoreFirst);

                // output parameters
                String treeFile, mappingFile;
                properties.TryGetValue("treeFile", out treeFile);
                properties.TryGetValue("mappingFile", out mappingFile);
                
                                
                if (ignoreFirst.Equals("true"))
                {
                    FileUtilities.GetFileData(dataFile, out ids, out data,
                        colSeperator.ToCharArray(), true);
                }
                else if (ignoreFirst.Equals("false"))
                {
                    FileUtilities.GetFileData(dataFile, out ids, out data,
                     colSeperator.ToCharArray(), false);
                }
                else
                {
                    throw new Exception();
                }
                
                tree.InitializeTree(ids, data);
                bool append = false;
                if (MPI.Communicator.world.Rank != 0)
                {
                    MPI.Communicator.world.Receive<int>(MPI.Communicator.world.Rank - 1, 0);
                    append = true;
                }
                tree.SaveToFile(treeFile, mappingFile, append);
                if (MPI.Communicator.world.Rank != MPI.Communicator.world.Size - 1)
                {
                    MPI.Communicator.world.Send<int>(0, MPI.Communicator.world.Rank + 1, 0);
                }
            }
            else if (io.Equals("db"))
            {
                String dataDb, dataTable, treeTable, treeDb;
                properties.TryGetValue("dataDb", out dataDb);
                properties.TryGetValue("dataTable", out dataTable);
                properties.TryGetValue("treeDb", out treeDb);
                properties.TryGetValue("treeTable", out treeTable);
                DatabaseUtilities.GetTableData(dataDb, dataTable, out ids, out data);
                tree.InitializeTree(ids, data);
                tree.SaveToDb(dataDb, dataTable, treeDb, treeTable);
            }
            else
            {
                throw new Exception();
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

        private static void ReadPropertiesFile(String fileName)
        {
            properties = new Dictionary<string, string>(13);
            if (!File.Exists(fileName))
            {
                throw new FileNotFoundException();
            }
            using (StreamReader sr = File.OpenText(fileName))
            {
                String input;
                while ((input = sr.ReadLine()) != null)
                {
                    
                    if (!input.StartsWith("--"))
                    {
                        String[] vals = input.Split(new char[] { '=' });
                        properties.Add(vals[0],vals[1].Substring(0,vals[1].Length-1));
                    }
                }
                sr.Close();
            }
        }

        private static String GetPropertyValue(String name)
        {
            String value;
            properties.TryGetValue(name, out value);
            return value;
        }
    }
}
