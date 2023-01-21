using System.IO;
using System.Windows.Forms;

namespace SARS_classification_app
{
    public static class Constants
    {
        public static readonly string PYTHON_PATH = @Properties.Settings.Default.PythonPath;
        public static readonly string DATA_FOLDER = Path.GetDirectoryName(Application.ExecutablePath) + @"\Process";
        public static readonly string SCRIPT_PATH = string.Format(@"""{0}\SARS_ANALYZE.py""", DATA_FOLDER);
        public static readonly string POST_CALC_PATH = string.Format(@"""{0}\post_calc.py""", DATA_FOLDER);
        public static readonly string INSTALL_PACKAGE = string.Format(@"-m pip install -r ""{0}\packages.txt""", DATA_FOLDER);

    }
}
