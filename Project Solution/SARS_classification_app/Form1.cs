#region "Directives"
using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Windows.Forms;
#endregion

namespace SARS_classification_app
{
    /// <summary>
    /// Form Class Controller.
    /// </summary>
    public partial class Form1 : Form
    {
        #region "Properties"

        private string filePath;
        private bool isDataLoaded;
        private Thread thread2 = null;
        private CheckBox checkedNgram;
        private CheckBox checkedMode;
        private int Ngrams = 3;
        private int mode = 0;

        #endregion

        #region "Ctor"

        /// <summary>
        /// Ctor for Form1
        /// </summary>
        public Form1()
        {

            InitializeComponent();
            FormBorderStyle = FormBorderStyle.FixedSingle;
            MaximizeBox = false;
            checkCSVFilesExistance();
            PostAnalyze_Bn.Enabled = IsPostAnalyzeReady();
        }

        #endregion

        #region "Buttons"

        /// <summary>
        /// Open file dialog to get data file path.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void LoadData_Bn_Click(object sender, EventArgs e)
        {
            using (OpenFileDialog openFileDialog = new OpenFileDialog())
            {
                openFileDialog.InitialDirectory = Constants.DATA_FOLDER;
                openFileDialog.Filter = "Text|*.txt";
                openFileDialog.FilterIndex = 1;
                openFileDialog.RestoreDirectory = true;

                if (openFileDialog.ShowDialog() == DialogResult.OK)
                {
                    //Get the path of specified file
                    filePath = openFileDialog.FileName;
                    if (filePath.Contains("data"))
                    {
                        checkBox1.Checked = true;
                        LoadData_Bn.Enabled = false;
                        isDataLoaded = true;
                    }
                    else
                    {
                        isDataLoaded = false;
                        LoadData_Bn.Enabled = true;
                        MessageBox.Show("Wrong sequence file, name incorrect", "ERROR", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    }
                }
            }
        }

        /// <summary>
        /// Running Train & analyze python script in another thread
        /// with Arguments: <mode> <Ngrams> <filePath> 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void TrainAndAnalyze_Bn_Click(object sender, EventArgs e)
        {
            if (checkedNgram != null && checkedMode != null)
            {
                checkedNgram.BackColor = System.Drawing.Color.Green;
                checkedNgram.Enabled = false;
                checkedNgram = null;
                checkedMode.BackColor = System.Drawing.Color.Green;
                label3.Text = "PROCESSING";
                label3.ForeColor = System.Drawing.Color.Tomato;
                thread2 = new Thread(() =>
                {
                    var path = filePath.Replace(" ", "###");
                    var args = string.Format("{0} {1} {2}", mode, Ngrams, path);
                    RunTrainAndAnalyze(args);
                });

                thread2.Start();
                if(IsPostAnalyzeReady())
                {
                    PostAnalyze_Bn.Enabled = true;
                }
                else
                {
                    PostAnalyze_Bn.Enabled = false;
                }
            }
            else
            {
                MessageBox.Show("Please choose Ngram and Mode", "ERROR", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }

        }

        /// <summary>
        /// Running post analyze python script in another thread.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void PostAnalyze_Bn_Click(object sender, EventArgs e)
        {
            if (IsPostAnalyzeReady())
            {
                var thread3 = new Thread(() =>
                {
                    RunPostCalc();
                });

                thread3.Start();
            }
            else
            {
                MessageBox.Show("Please make sure that you have the following files\nin the Process folder:" +
                    "\n\n1. dist_mat_3.csv" +
                    "\n\n2. dist_mat_4.csv " +
                    "\n\n3. dist_mat_5.csv", "ERROR", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        /// <summary>
        /// open Result Directory.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Result_Bn_Click(object sender, EventArgs e)
        {
            var path = Constants.DATA_FOLDER;
            Process.Start(path);
        }

        /// <summary>
        /// User instruction with the flow and notes.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void UserHelp_Bn_Click(object sender, EventArgs e)
        {
            var title = "User Instructions:";
            var userInstructions =
                "1.Load sequences file - the file name should be 'data'." +
                "\n\n2.Choose the desired Ngram for model training.: 3/4/5. " +
                "\n\n3.Choose mode for model training: Full run or Recovery. " +
                 "\n\n4.Click on Train && Analyze button to start processing." +
                 "\n\n5.Click on Post Analyze button." +
                 "\n\n6.Click on Results button to open folder results." +
                 "\n\n NOTE: " +
                "\n\n*Recovery mode will continue the training in the same" +
                "\n spot if where the training processing crushed." +
                "\n\nPost Analyze will only be possible when all Ngrams csv files" +
                "\n will be in the process folder after running for each Ngrams" +
                "\n\n*You should change python.exe path in the config file" +
                "\n that is located in the same folder.";
            MessageBox.Show(userInstructions, title, MessageBoxButtons.OK, MessageBoxIcon.Information);
        }

        #endregion

        #region "Ngrams"

        /// <summary>
        /// Ngram = 3 Checked
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void checkBox3_CheckedChanged(object sender, EventArgs e)
        {
            checkBox4.Checked = false;
            checkBox5.Checked = false;
            if (checkBox3.CheckState == CheckState.Checked)
            {
                Ngrams = 3;
                checkedNgram = checkBox3;
            }
            else
            {
                checkedNgram = null;
            }
            if (checkedNgram != null && checkedMode != null)
            {
                TrainAndAnalyze_Bn.Enabled = true;
            }
        }

        /// <summary>
        /// Ngram = 4 Checked
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void checkBox4_CheckedChanged(object sender, EventArgs e)
        {
            checkBox3.Checked = false;
            checkBox5.Checked = false;
            if (checkBox4.CheckState == CheckState.Checked)
            {
                Ngrams = 4;
                checkedNgram = checkBox4;
            }
            else
            {
                checkedNgram = null;
            }
            if (checkedNgram != null && checkedMode != null)
            {
                TrainAndAnalyze_Bn.Enabled = true;
            }
        }

        /// <summary>
        /// Ngram = 5 Checked
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void checkBox5_CheckedChanged(object sender, EventArgs e)
        {
            checkBox4.Checked = false;
            checkBox3.Checked = false;
            if (checkBox5.CheckState == CheckState.Checked)
            {

                Ngrams = 5;
                checkedNgram = checkBox5;
            }
            else
            {
                checkedNgram = null;
            }
            if (checkedNgram != null && checkedMode != null)
            {
                TrainAndAnalyze_Bn.Enabled = true;
            }

        }

        #endregion

        #region "Modes"

        /// <summary>
        /// mode = "Full run" checked.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void mode_FR_CheckedChanged(object sender, EventArgs e)
        {
            mode_Rec.Checked = false;
            if (mode_FR.CheckState == CheckState.Checked)
            {
                mode = 0;
                checkedMode = mode_FR;
            }
            else
            {
                checkedMode = null;
            }

            if (checkedNgram != null && checkedMode != null && isDataLoaded)
            {
                TrainAndAnalyze_Bn.Enabled = true;
            }
        }

        /// <summary>
        /// mode = "Full run" checked.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void mode_Rec_CheckedChanged(object sender, EventArgs e)
        {
            mode_FR.Checked = false;
            if (mode_Rec.CheckState == CheckState.Checked)
            {
                mode = 2;
                checkedMode = mode_Rec;
            }
            else
            {
                checkedMode = null;
            }
            if (checkedNgram != null && checkedMode != null && isDataLoaded)
            {
                TrainAndAnalyze_Bn.Enabled = true;
            }
        }

        #endregion

        #region Python Scripts

        /// <summary>
        /// Running cmd process, installing required packages for python 
        /// and running "Train & Test " python script with args : <Mode> <Ngram> <filePath>
        /// </summary>
        /// <param name="args"></param>
        private void RunTrainAndAnalyze(string args)
        {
            using (Process p = new Process())
            {
                p.StartInfo.FileName = @"cmd.exe";
                p.StartInfo.WindowStyle = ProcessWindowStyle.Normal;
                p.StartInfo.UseShellExecute = false;
                p.StartInfo.RedirectStandardOutput = true;
                p.StartInfo.RedirectStandardInput = true;
                p.StartInfo.CreateNoWindow = false;
                p.Start();
                p.StandardInput.WriteLine($"cd {Constants.DATA_FOLDER}");
                p.StandardInput.WriteLine(string.Format("{0} {1}", Constants.PYTHON_PATH, Constants.INSTALL_PACKAGE));
                p.StandardInput.WriteLine(string.Format("{0} {1} {2}", Constants.PYTHON_PATH, Constants.SCRIPT_PATH, args));
                string output = p.StandardOutput.ReadToEnd();
                p.WaitForExit();
            }
            label3.Text = "FINISHED";
            label3.ForeColor = System.Drawing.Color.Green;
            Result_Bn.Enabled = true;
            PostAnalyze_Bn.Enabled = IsPostAnalyzeReady();
        }

        /// <summary>
        /// Run post calc python script via Cmd process.
        /// </summary>
        private void RunPostCalc()
        {
            using (Process p = new Process())
            {
                p.StartInfo.FileName = @"cmd.exe";
                p.StartInfo.WindowStyle = ProcessWindowStyle.Normal;
                p.StartInfo.UseShellExecute = false;
                p.StartInfo.RedirectStandardOutput = true;
                p.StartInfo.RedirectStandardInput = true;
                p.StartInfo.CreateNoWindow = false;
                p.Start();
                p.StandardInput.WriteLine($"cd {Constants.DATA_FOLDER}");
                p.StandardInput.WriteLine(string.Format("{0} {1}", Constants.PYTHON_PATH, Constants.POST_CALC_PATH));

                string output = p.StandardOutput.ReadToEnd();
                p.WaitForExit();
                Result_Bn.Enabled = Enabled;
            }
        }

        #endregion

        #region "Help Methods"

        /// <summary>
        /// Checks if Post analyze ready, when CSV files for Ngrams = 3,4,5 is Exist.
        /// </summary>
        /// <returns>boolean</returns>
        private bool IsPostAnalyzeReady()
        {
            bool isExist3, isExist4, isExist5;
            if (File.Exists(Constants.DATA_FOLDER + "\\dist_mat_3.csv"))
            {
                isExist3 = true;
            }
            else
            {
                isExist3 = false;
            }
            if (File.Exists(Constants.DATA_FOLDER + "\\dist_mat_4.csv"))
            {
                isExist4 = true;
            }
            else
            {
                isExist4 = false;
            }
            if (File.Exists(Constants.DATA_FOLDER + "\\dist_mat_5.csv"))
            {
                isExist5 = true;
            }
            else
            {
                isExist5 = false;
            }
            return isExist3 && isExist4 && isExist5;
        }

        /// <summary>
        /// Checks if CSV file dist_mat_X.csv is already exist for Ngram = X
        /// means run already finished for Ngram = X.
        /// </summary>
        private void checkCSVFilesExistance()
        {
            if (File.Exists(Constants.DATA_FOLDER + "\\dist_mat_3.csv"))
            {
                checkBox3.Enabled = false;
                checkBox3.Checked = true;
                checkBox3.BackColor = System.Drawing.Color.Green;
            }
            if (File.Exists(Constants.DATA_FOLDER + "\\dist_mat_4.csv"))
            {
                checkBox4.Enabled = false;
                checkBox4.Checked = true;
                checkBox4.BackColor = System.Drawing.Color.Green;
            }
            if (File.Exists(Constants.DATA_FOLDER + "\\dist_mat_5.csv"))
            {
                checkBox5.Enabled = false;
                checkBox5.Checked = true;
                checkBox5.BackColor = System.Drawing.Color.Green;
            }
        }

        #endregion
    }
}