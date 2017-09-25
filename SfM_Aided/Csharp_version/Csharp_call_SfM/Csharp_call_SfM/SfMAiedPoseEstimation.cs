using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using System.Runtime.InteropServices;

namespace Csharp_call_SfM
{
    class SfMAiedPoseEstimation
    {
        //- The common settings of exe paths
        public static string OPENMVG_SFM_BIN = "H:/projects/SLAM/sfm/openMVG_develop/src/release/Windows-AMD64-Release/Release";
        public static string OPENMVG_Localization_BIN = OPENMVG_SFM_BIN;
        public static string WORK_DIR = "H:/projects/SLAM/pose_estimation/SfM_Aided/";
        public static string OPENMVG_SFM_MINE_BIN = WORK_DIR + "build/SfM/Release";
        public static string OPENMVG_Geodesy_MINE_BIN = WORK_DIR + "build/Geodesy/Release";



        // [const char* as parameter in C++](https://msdn.microsoft.com/en-us/library/system.runtime.interopservices.callingconvention.aspx)
        // CharSet = Ansi, please take attention
        [DllImport("H:/projects/SLAM/pose_estimation/SfM_Aided/build/SfM/Release/relativePosePair.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        public static extern bool relative_pose_of_file(String input_file, String reference_im_name, String query_im_name, ref double thetaz, ref double thetay, ref double thetax, ref double x, ref double y, ref double z);


        static void runCmdArr(string[] cmd_arr)
        {
            string cmd = string.Join(" ", cmd_arr);
            Console.WriteLine(cmd);
            string output = "";
            CmdHelper.RunCmd(cmd, out output);
            System.Console.WriteLine(output);
        }





        static void incremental_SfM_pipeline(string dataset_dir, string output_dir, string K_value)
        {
            string init_dir = dataset_dir + "/sfm_init_data";
            string faked_gps_path = dataset_dir + "/fake_gps_file.txt";

            string matches_dir = output_dir + "/matches";
            string reconstruction_dir = output_dir + "/reconstruction_sequential";

            DirectoryInfo oDir_1 = new DirectoryInfo(Path.GetFullPath(matches_dir));
            if (!oDir_1.Exists)
            {
                oDir_1.Create();
            }
            DirectoryInfo oDir_2 = new DirectoryInfo(Path.GetFullPath(reconstruction_dir));
            if (!oDir_2.Exists)
            {
                oDir_2.Create();
            }

            System.Console.WriteLine("1. Intrinsics analysis");
            string[] cmd_arr_1 = {OPENMVG_SFM_MINE_BIN+"/SfMInit_ImageListing", "--imageDirectory", init_dir, "--outputDirectory", matches_dir, "--intrinsics", K_value};
            runCmdArr(cmd_arr_1);

            System.Console.WriteLine("1.x Geodesy, add position prior");
            string[] cmd_arr_1x = { OPENMVG_Geodesy_MINE_BIN + "/registration_faked_gps_position", "--input_file", matches_dir + "/sfm_data.json", "--output_file", matches_dir + "/sfm_data.json", "--faked_gps_path", faked_gps_path };
            runCmdArr(cmd_arr_1x);


            System.Console.WriteLine("2. Compute features");
            string[] cmd_arr_2 = { OPENMVG_SFM_BIN + "/openMVG_main_ComputeFeatures", "--input_file", matches_dir + "/sfm_data.json", "--outdir", matches_dir, "--describerMethod", "SIFT", "--numThreads", "7" }; // threads num is related to CPU
            runCmdArr(cmd_arr_2);


            System.Console.WriteLine("3. Compute matches");
            string[] cmd_arr_3 = { OPENMVG_SFM_BIN + "/openMVG_main_ComputeMatches", "--input_file", matches_dir + "/sfm_data.json", "--out_dir", matches_dir };
            //string[] cmd_arr_3 = { OPENMVG_SFM_BIN + "/openMVG_main_ComputeMatches", "--input_file", matches_dir + "/sfm_data.json", "--out_dir", matches_dir, "--geometric_model", "e", "--guided_matching", "1" };
            runCmdArr(cmd_arr_3);


            System.Console.WriteLine("4. Do Sequential/Incremental reconstruction");
            string[] cmd_arr_4 = { OPENMVG_SFM_BIN + "/openMVG_main_IncrementalSfM", "--input_file", matches_dir + "/sfm_data.json", "--matchdir", matches_dir, "--outdir", reconstruction_dir, "--refineIntrinsics", "NONE", "--prior_usage" };
            runCmdArr(cmd_arr_4);


            System.Console.WriteLine("4.1 Convert format for easy looking");
            // cvt_SfM_data(reconstruction_dir, reconstruction_dir, "sfm_data.bin")

            System.Console.WriteLine("4.2 Colorize Structure");// # no need to do this, to save time
            string[] cmd_arr_42 = { OPENMVG_SFM_BIN + "/openMVG_main_ComputeSfM_DataColor", "-i", reconstruction_dir + "/sfm_data.bin", "-o", reconstruction_dir + "/colorized_incremental_SfM.ply" };
            runCmdArr(cmd_arr_42);

            // optional, compute final valid structure from the known camera poses, a refine part
            // Note the matches.e.bin or matches.f.bin
            System.Console.WriteLine("5. Structure from Known Poses (robust triangulation)");
            string[] cmd_arr_5 = { OPENMVG_SFM_BIN + "/openMVG_main_ComputeStructureFromKnownPoses", "--input_file", reconstruction_dir + "/sfm_data.bin", "--match_dir", matches_dir, "--output_file", reconstruction_dir+"/robust.bin" };
            runCmdArr(cmd_arr_5);



            System.Console.WriteLine("5.1 Convert format for easy looking");
            //cvt_SfM_data(reconstruction_dir, reconstruction_dir, "robust.bin")

            System.Console.WriteLine("5.2 Colorize Structure");
            string[] cmd_arr_52 = { OPENMVG_SFM_BIN + "/openMVG_main_ComputeSfM_DataColor", "-i", reconstruction_dir + "/robust.bin", "-o", reconstruction_dir+"/robust_colorized.ply"};
            runCmdArr(cmd_arr_52);
        }




        static void localization_pipeline(string dataset_dir, string output_dir,
          string incremental_SfM_data_name,
          string reference_im_name = "reference_2.jpg",
          string query_im_name = "query_1.jpg"
          )
        {
            string query_dir = dataset_dir + "/sfm_query_data";

            string matches_dir = output_dir + "/matches";
            string reconstruction_dir = output_dir + "/reconstruction_sequential";
            string localization_dir = output_dir + "/Localization";

            string query_im_path = query_dir + "/" + query_im_name;

            System.Console.WriteLine("Using dataset dir:\n{0}", dataset_dir);
            System.Console.WriteLine("      output_dir :\n{0}", output_dir);
            System.Console.WriteLine("      localization_dir:\n{0}", localization_dir);

            DirectoryInfo oDir_1 = new DirectoryInfo(Path.GetFullPath(matches_dir));
            if (!oDir_1.Exists)
            {
                System.Console.WriteLine("seems we have not initialized SfM");
                return;
            }
            DirectoryInfo oDir_2 = new DirectoryInfo(Path.GetFullPath(localization_dir));
            if (!oDir_2.Exists)
            {
                oDir_2.Create();
            }


            //##NOTE: the `-s –single_intrinsics` is rather important, it will not try to BA the intrinsics, leave our methods more stable
            System.Console.WriteLine("1. Localization ..");
            string[] cmd_arr_1 = { OPENMVG_Localization_BIN + "/openMVG_main_SfM_Localization", "--input_file", reconstruction_dir + "/" + incremental_SfM_data_name, "--match_dir", matches_dir, "--out_dir", localization_dir, "--match_out_dir", localization_dir, "--query_image_dir", query_im_path, "--single_intrinsics" };
            runCmdArr(cmd_arr_1);

            System.Console.WriteLine("2. Calc relative pose of {0}, regrding to reference image {1}", query_im_name, reference_im_name);
            string[] cmd_arr_2 = { OPENMVG_SFM_MINE_BIN + "/relativePosePair_test", "--input_file", localization_dir + "/sfm_data_expanded.json", "--reference_im_name", reference_im_name, "--query_im_name", query_im_name };
            runCmdArr(cmd_arr_2);

            System.Console.WriteLine("Run in C#");
            double thetaz = 0.0, thetay = 0.0, thetax = 0.0, x = 0.0, y = 0.0, z = 0.0;
            bool rtn = relative_pose_of_file(localization_dir + "/sfm_data_expanded.json", reference_im_name, query_im_name, ref thetaz, ref thetay, ref thetax, ref x, ref y, ref z);
            if (rtn)
            {
                System.Console.WriteLine("We got it");
                double[] six_dof = { thetaz, thetay, thetax, x, y, z };
                System.Console.WriteLine(String.Join(" ", six_dof));
            }
            else
            {
                System.Console.WriteLine("Seems something wrong");
            }

        }


        static void Main(string[] args)
        {

            //string dataset_dir = WORK_DIR + "dataset_cartoon_1_bmp/";
            //string output_dir = WORK_DIR + "sequential_cartoon_1_bmp_Csharp";
            //string K_value = "1444.29449;0.0;482.68264;0.0;1444.79783;319.3993;0.0;0.0;1"; // K for small images, bmp

            //string dataset_dir = WORK_DIR + "dataset_Marx_1/";
            //string output_dir = WORK_DIR + "sequential_Marx_1_jpg_Csharp";
            //string K_value = "8607.8639;0;2880.72115;0;8605.4303;1913.87935;0;0;1";// # K for big images, jpg

            string dataset_dir = WORK_DIR + "dataset_cartoon_1_less_pic/";
            string output_dir = WORK_DIR + "sequential_dataset_cartoon_1_less_jpg_Csharp";
            string K_value = "8607.8639;0;2880.72115;0;8605.4303;1913.87935;0;0;1";// # K for big images, jpg


            

            //incremental_SfM_pipeline(dataset_dir, output_dir, K_value);


            //localization_pipeline(dataset_dir, output_dir, "robust.bin", "reference_2.jpg",  "query_1.jpg");
            localization_pipeline(dataset_dir, output_dir, "robust.bin", "1.jpg", "query_4.jpg");

            Console.ReadKey(true);

        }


    }
}
