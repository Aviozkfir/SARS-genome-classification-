namespace SARS_classification_app
{
    partial class Form1
    {
        /// <summary>
        /// Required designer SARS_classification_app.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(Form1));
            this.TrainAndAnalyze_Bn = new System.Windows.Forms.Button();
            this.label1 = new System.Windows.Forms.Label();
            this.checkBox1 = new System.Windows.Forms.CheckBox();
            this.label5 = new System.Windows.Forms.Label();
            this.label6 = new System.Windows.Forms.Label();
            this.LoadData_Bn = new System.Windows.Forms.Button();
            this.label2 = new System.Windows.Forms.Label();
            this.label8 = new System.Windows.Forms.Label();
            this.label9 = new System.Windows.Forms.Label();
            this.label11 = new System.Windows.Forms.Label();
            this.Result_Bn = new System.Windows.Forms.Button();
            this.label13 = new System.Windows.Forms.Label();
            this.panel1 = new System.Windows.Forms.Panel();
            this.panel2 = new System.Windows.Forms.Panel();
            this.UserHelp_Bn = new System.Windows.Forms.Button();
            this.label4 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.label7 = new System.Windows.Forms.Label();
            this.label10 = new System.Windows.Forms.Label();
            this.label12 = new System.Windows.Forms.Label();
            this.label14 = new System.Windows.Forms.Label();
            this.checkBox2 = new System.Windows.Forms.CheckBox();
            this.checkBox3 = new System.Windows.Forms.CheckBox();
            this.checkBox4 = new System.Windows.Forms.CheckBox();
            this.checkBox5 = new System.Windows.Forms.CheckBox();
            this.mode_FR = new System.Windows.Forms.CheckBox();
            this.mode_Rec = new System.Windows.Forms.CheckBox();
            this.label15 = new System.Windows.Forms.Label();
            this.PostAnalyze_Bn = new System.Windows.Forms.Button();
            this.label16 = new System.Windows.Forms.Label();
            this.panel1.SuspendLayout();
            this.panel2.SuspendLayout();
            this.SuspendLayout();
            // 
            // TrainAndAnalyze_Bn
            // 
            this.TrainAndAnalyze_Bn.BackColor = System.Drawing.Color.WhiteSmoke;
            this.TrainAndAnalyze_Bn.Enabled = false;
            this.TrainAndAnalyze_Bn.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.TrainAndAnalyze_Bn.Font = new System.Drawing.Font("Century Gothic", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.TrainAndAnalyze_Bn.Location = new System.Drawing.Point(158, 324);
            this.TrainAndAnalyze_Bn.Name = "TrainAndAnalyze_Bn";
            this.TrainAndAnalyze_Bn.Size = new System.Drawing.Size(135, 33);
            this.TrainAndAnalyze_Bn.TabIndex = 0;
            this.TrainAndAnalyze_Bn.Text = "Train && Analyze";
            this.TrainAndAnalyze_Bn.UseVisualStyleBackColor = false;
            this.TrainAndAnalyze_Bn.Click += new System.EventHandler(this.TrainAndAnalyze_Bn_Click);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Font = new System.Drawing.Font("Microsoft Sans Serif", 20.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label1.ForeColor = System.Drawing.SystemColors.ButtonHighlight;
            this.label1.Location = new System.Drawing.Point(92, 9);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(310, 62);
            this.label1.TabIndex = 2;
            this.label1.Text = "COVID-19 \r\ngenomes classification";
            this.label1.TextAlign = System.Drawing.ContentAlignment.TopCenter;
            // 
            // checkBox1
            // 
            this.checkBox1.AutoSize = true;
            this.checkBox1.Enabled = false;
            this.checkBox1.Location = new System.Drawing.Point(388, 126);
            this.checkBox1.Name = "checkBox1";
            this.checkBox1.Size = new System.Drawing.Size(15, 14);
            this.checkBox1.TabIndex = 8;
            this.checkBox1.UseVisualStyleBackColor = true;
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Font = new System.Drawing.Font("Century Gothic", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label5.ForeColor = System.Drawing.SystemColors.ActiveCaptionText;
            this.label5.Location = new System.Drawing.Point(12, 125);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(63, 21);
            this.label5.TabIndex = 9;
            this.label5.Text = "Step 1:";
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Font = new System.Drawing.Font("Century Gothic", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label6.ForeColor = System.Drawing.SystemColors.ActiveCaptionText;
            this.label6.Location = new System.Drawing.Point(12, 193);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(63, 21);
            this.label6.TabIndex = 10;
            this.label6.Text = "Step 2:";
            // 
            // LoadData_Bn
            // 
            this.LoadData_Bn.BackColor = System.Drawing.Color.WhiteSmoke;
            this.LoadData_Bn.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.LoadData_Bn.Font = new System.Drawing.Font("Century Gothic", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.LoadData_Bn.Location = new System.Drawing.Point(277, 119);
            this.LoadData_Bn.Name = "LoadData_Bn";
            this.LoadData_Bn.Size = new System.Drawing.Size(87, 27);
            this.LoadData_Bn.TabIndex = 11;
            this.LoadData_Bn.Text = "Click here";
            this.LoadData_Bn.UseVisualStyleBackColor = false;
            this.LoadData_Bn.Click += new System.EventHandler(this.LoadData_Bn_Click);
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Font = new System.Drawing.Font("Century Gothic", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label2.ForeColor = System.Drawing.SystemColors.ActiveCaptionText;
            this.label2.Location = new System.Drawing.Point(3, 105);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(0, 17);
            this.label2.TabIndex = 5;
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(-3, 231);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(478, 13);
            this.label8.TabIndex = 13;
            this.label8.Text = "---------------------------------------------------------------------------------" +
    "----------------------------------------------------------------------------\r\n";
            // 
            // label9
            // 
            this.label9.AutoSize = true;
            this.label9.Location = new System.Drawing.Point(-3, 160);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(484, 13);
            this.label9.TabIndex = 14;
            this.label9.Text = "---------------------------------------------------------------------------------" +
    "------------------------------------------------------------------------------\r\n" +
    "";
            // 
            // label11
            // 
            this.label11.AutoSize = true;
            this.label11.Font = new System.Drawing.Font("Century Gothic", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label11.ForeColor = System.Drawing.SystemColors.ActiveCaptionText;
            this.label11.Location = new System.Drawing.Point(94, 125);
            this.label11.Name = "label11";
            this.label11.Size = new System.Drawing.Size(161, 21);
            this.label11.TabIndex = 16;
            this.label11.Text = "Load sequences file";
            // 
            // Result_Bn
            // 
            this.Result_Bn.BackColor = System.Drawing.Color.WhiteSmoke;
            this.Result_Bn.Enabled = false;
            this.Result_Bn.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.Result_Bn.Font = new System.Drawing.Font("Century Gothic", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.Result_Bn.Location = new System.Drawing.Point(158, 9);
            this.Result_Bn.Name = "Result_Bn";
            this.Result_Bn.Size = new System.Drawing.Size(135, 33);
            this.Result_Bn.TabIndex = 18;
            this.Result_Bn.Text = "Results";
            this.Result_Bn.UseVisualStyleBackColor = false;
            this.Result_Bn.Click += new System.EventHandler(this.Result_Bn_Click);
            // 
            // label13
            // 
            this.label13.AutoSize = true;
            this.label13.Font = new System.Drawing.Font("Century Gothic", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label13.ForeColor = System.Drawing.Color.AliceBlue;
            this.label13.Location = new System.Drawing.Point(12, 18);
            this.label13.Name = "label13";
            this.label13.Size = new System.Drawing.Size(108, 21);
            this.label13.TabIndex = 20;
            this.label13.Text = "Version: 0.0.1";
            // 
            // panel1
            // 
            this.panel1.BackColor = System.Drawing.Color.DarkSeaGreen;
            this.panel1.Controls.Add(this.label1);
            this.panel1.Controls.Add(this.label2);
            this.panel1.Dock = System.Windows.Forms.DockStyle.Top;
            this.panel1.Location = new System.Drawing.Point(0, 0);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(467, 100);
            this.panel1.TabIndex = 21;
            // 
            // panel2
            // 
            this.panel2.BackColor = System.Drawing.Color.DarkSeaGreen;
            this.panel2.Controls.Add(this.UserHelp_Bn);
            this.panel2.Controls.Add(this.label13);
            this.panel2.Controls.Add(this.Result_Bn);
            this.panel2.Dock = System.Windows.Forms.DockStyle.Bottom;
            this.panel2.Location = new System.Drawing.Point(0, 446);
            this.panel2.Name = "panel2";
            this.panel2.Size = new System.Drawing.Size(467, 54);
            this.panel2.TabIndex = 0;
            // 
            // UserHelp_Bn
            // 
            this.UserHelp_Bn.BackColor = System.Drawing.Color.WhiteSmoke;
            this.UserHelp_Bn.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.UserHelp_Bn.Font = new System.Drawing.Font("Century Gothic", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.UserHelp_Bn.Location = new System.Drawing.Point(320, 9);
            this.UserHelp_Bn.Name = "UserHelp_Bn";
            this.UserHelp_Bn.Size = new System.Drawing.Size(135, 33);
            this.UserHelp_Bn.TabIndex = 21;
            this.UserHelp_Bn.Text = "User Help";
            this.UserHelp_Bn.UseVisualStyleBackColor = false;
            this.UserHelp_Bn.Click += new System.EventHandler(this.UserHelp_Bn_Click);
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Font = new System.Drawing.Font("Century Gothic", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label4.ForeColor = System.Drawing.SystemColors.ActiveCaptionText;
            this.label4.Location = new System.Drawing.Point(12, 263);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(63, 21);
            this.label4.TabIndex = 22;
            this.label4.Text = "Step 3:";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Font = new System.Drawing.Font("Century Gothic", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label3.ForeColor = System.Drawing.SystemColors.ActiveCaptionText;
            this.label3.Location = new System.Drawing.Point(357, 328);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(65, 21);
            this.label3.TabIndex = 6;
            this.label3.Text = "READY";
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(-3, 299);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(478, 13);
            this.label7.TabIndex = 23;
            this.label7.Text = "---------------------------------------------------------------------------------" +
    "----------------------------------------------------------------------------\r\n";
            // 
            // label10
            // 
            this.label10.AutoSize = true;
            this.label10.Font = new System.Drawing.Font("Century Gothic", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label10.ForeColor = System.Drawing.SystemColors.ActiveCaptionText;
            this.label10.Location = new System.Drawing.Point(12, 328);
            this.label10.Name = "label10";
            this.label10.Size = new System.Drawing.Size(63, 21);
            this.label10.TabIndex = 24;
            this.label10.Text = "Step 4:";
            // 
            // label12
            // 
            this.label12.AutoSize = true;
            this.label12.Font = new System.Drawing.Font("Century Gothic", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label12.ForeColor = System.Drawing.SystemColors.ActiveCaptionText;
            this.label12.Location = new System.Drawing.Point(94, 193);
            this.label12.Name = "label12";
            this.label12.Size = new System.Drawing.Size(133, 21);
            this.label12.TabIndex = 25;
            this.label12.Text = "Choose Ngrams";
            // 
            // label14
            // 
            this.label14.AutoSize = true;
            this.label14.Font = new System.Drawing.Font("Century Gothic", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label14.ForeColor = System.Drawing.SystemColors.ActiveCaptionText;
            this.label14.Location = new System.Drawing.Point(94, 263);
            this.label14.Name = "label14";
            this.label14.Size = new System.Drawing.Size(119, 21);
            this.label14.TabIndex = 27;
            this.label14.Text = "Choose Mode";
            // 
            // checkBox2
            // 
            this.checkBox2.AutoSize = true;
            this.checkBox2.Location = new System.Drawing.Point(542, 390);
            this.checkBox2.Name = "checkBox2";
            this.checkBox2.Size = new System.Drawing.Size(80, 17);
            this.checkBox2.TabIndex = 29;
            this.checkBox2.Text = "checkBox2";
            this.checkBox2.UseVisualStyleBackColor = true;
            // 
            // checkBox3
            // 
            this.checkBox3.AutoSize = true;
            this.checkBox3.Location = new System.Drawing.Point(261, 193);
            this.checkBox3.Name = "checkBox3";
            this.checkBox3.Size = new System.Drawing.Size(32, 17);
            this.checkBox3.TabIndex = 30;
            this.checkBox3.Text = "3";
            this.checkBox3.UseVisualStyleBackColor = true;
            this.checkBox3.CheckedChanged += new System.EventHandler(this.checkBox3_CheckedChanged);
            // 
            // checkBox4
            // 
            this.checkBox4.AutoSize = true;
            this.checkBox4.Location = new System.Drawing.Point(320, 193);
            this.checkBox4.Name = "checkBox4";
            this.checkBox4.Size = new System.Drawing.Size(32, 17);
            this.checkBox4.TabIndex = 31;
            this.checkBox4.Text = "4";
            this.checkBox4.UseVisualStyleBackColor = true;
            this.checkBox4.CheckedChanged += new System.EventHandler(this.checkBox4_CheckedChanged);
            // 
            // checkBox5
            // 
            this.checkBox5.AutoSize = true;
            this.checkBox5.Location = new System.Drawing.Point(371, 193);
            this.checkBox5.Name = "checkBox5";
            this.checkBox5.Size = new System.Drawing.Size(32, 17);
            this.checkBox5.TabIndex = 32;
            this.checkBox5.Text = "5";
            this.checkBox5.UseVisualStyleBackColor = true;
            this.checkBox5.CheckedChanged += new System.EventHandler(this.checkBox5_CheckedChanged);
            // 
            // mode_FR
            // 
            this.mode_FR.AutoSize = true;
            this.mode_FR.Location = new System.Drawing.Point(261, 263);
            this.mode_FR.Name = "mode_FR";
            this.mode_FR.Size = new System.Drawing.Size(60, 17);
            this.mode_FR.TabIndex = 33;
            this.mode_FR.Text = "Full run";
            this.mode_FR.UseVisualStyleBackColor = true;
            this.mode_FR.CheckedChanged += new System.EventHandler(this.mode_FR_CheckedChanged);
            // 
            // mode_Rec
            // 
            this.mode_Rec.AutoSize = true;
            this.mode_Rec.Location = new System.Drawing.Point(332, 263);
            this.mode_Rec.Name = "mode_Rec";
            this.mode_Rec.Size = new System.Drawing.Size(67, 17);
            this.mode_Rec.TabIndex = 34;
            this.mode_Rec.Text = "Recover";
            this.mode_Rec.UseVisualStyleBackColor = true;
            this.mode_Rec.CheckedChanged += new System.EventHandler(this.mode_Rec_CheckedChanged);
            // 
            // label15
            // 
            this.label15.AutoSize = true;
            this.label15.Location = new System.Drawing.Point(-11, 369);
            this.label15.Name = "label15";
            this.label15.Size = new System.Drawing.Size(478, 13);
            this.label15.TabIndex = 35;
            this.label15.Text = "---------------------------------------------------------------------------------" +
    "----------------------------------------------------------------------------\r\n";
            // 
            // PostAnalyze_Bn
            // 
            this.PostAnalyze_Bn.BackColor = System.Drawing.Color.WhiteSmoke;
            this.PostAnalyze_Bn.Enabled = false;
            this.PostAnalyze_Bn.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.PostAnalyze_Bn.Font = new System.Drawing.Font("Century Gothic", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.PostAnalyze_Bn.Location = new System.Drawing.Point(158, 395);
            this.PostAnalyze_Bn.Name = "PostAnalyze_Bn";
            this.PostAnalyze_Bn.Size = new System.Drawing.Size(135, 33);
            this.PostAnalyze_Bn.TabIndex = 36;
            this.PostAnalyze_Bn.Text = "Post Analyze";
            this.PostAnalyze_Bn.UseVisualStyleBackColor = false;
            this.PostAnalyze_Bn.Click += new System.EventHandler(this.PostAnalyze_Bn_Click);
            // 
            // label16
            // 
            this.label16.AutoSize = true;
            this.label16.Font = new System.Drawing.Font("Century Gothic", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label16.ForeColor = System.Drawing.SystemColors.ActiveCaptionText;
            this.label16.Location = new System.Drawing.Point(12, 399);
            this.label16.Name = "label16";
            this.label16.Size = new System.Drawing.Size(63, 21);
            this.label16.TabIndex = 37;
            this.label16.Text = "Step 5:";
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackColor = System.Drawing.Color.Snow;
            this.ClientSize = new System.Drawing.Size(467, 500);
            this.Controls.Add(this.label16);
            this.Controls.Add(this.PostAnalyze_Bn);
            this.Controls.Add(this.label15);
            this.Controls.Add(this.mode_Rec);
            this.Controls.Add(this.mode_FR);
            this.Controls.Add(this.checkBox5);
            this.Controls.Add(this.checkBox4);
            this.Controls.Add(this.checkBox3);
            this.Controls.Add(this.checkBox2);
            this.Controls.Add(this.label14);
            this.Controls.Add(this.label12);
            this.Controls.Add(this.label10);
            this.Controls.Add(this.label7);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.panel2);
            this.Controls.Add(this.TrainAndAnalyze_Bn);
            this.Controls.Add(this.panel1);
            this.Controls.Add(this.label11);
            this.Controls.Add(this.label9);
            this.Controls.Add(this.label8);
            this.Controls.Add(this.LoadData_Bn);
            this.Controls.Add(this.label6);
            this.Controls.Add(this.label5);
            this.Controls.Add(this.checkBox1);
            this.Controls.Add(this.label3);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.None;
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.Name = "Form1";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "COVID-19 Analyzing";
            this.panel1.ResumeLayout(false);
            this.panel1.PerformLayout();
            this.panel2.ResumeLayout(false);
            this.panel2.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button TrainAndAnalyze_Bn;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.CheckBox checkBox1;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.Button LoadData_Bn;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.Label label9;
        private System.Windows.Forms.Label label11;
        private System.Windows.Forms.Button Result_Bn;
        private System.Windows.Forms.Label label13;
        private System.Windows.Forms.Panel panel1;
        private System.Windows.Forms.Panel panel2;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Button UserHelp_Bn;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.Label label10;
        private System.Windows.Forms.Label label12;
        private System.Windows.Forms.Label label14;
        private System.Windows.Forms.CheckBox checkBox2;
        private System.Windows.Forms.CheckBox checkBox3;
        private System.Windows.Forms.CheckBox checkBox4;
        private System.Windows.Forms.CheckBox checkBox5;
        private System.Windows.Forms.CheckBox mode_FR;
        private System.Windows.Forms.CheckBox mode_Rec;
        private System.Windows.Forms.Label label15;
        private System.Windows.Forms.Button PostAnalyze_Bn;
        private System.Windows.Forms.Label label16;
    }
}

