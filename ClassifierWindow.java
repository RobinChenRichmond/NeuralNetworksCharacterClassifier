
/**
 * The purpose of this class is to act as the main control
 * window for performing a classification task.
 * Among functions this class should perform are
 * 
 * 1. Loading a training file (a file filled with training vectors in the format
 * we've been discussing in lecture).
 * 
 * 2. Performing backpropagation with that training file, creating
 * a weight matrix for the neural network.
 * 
 * 3. Save a weight matrix that has been previously derived via backpropagation.
 * 
 * 4. Load a previously saved weight matrix.
 * 
 * 5. Perform classification given an input vector and a weight matrix.  Classification
 * can be performed on characters drawn in the canvas or on vectors that are read from
 * an input file.
 * 
 * 
 * 
 * @author (your name) 
 * @version (a version number or a date)
 */

import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.event.*;
import java.awt.geom.*;
import java.util.*;
import java.io.*;
import Jama.Matrix;
import java.text.*;

public class ClassifierWindow extends WindowManager {

    // Stuff from original PaintWindow
    private static final Color BLACK = Color.BLACK;
    private static final Color WHITE = Color.WHITE;

    private final int NUM_ROWS = 20;    // # of rows in the layout for colored labels
    private final int NUM_COLS = 20;    // # of cols in the layout for colored labels
    private final int PAD      = 20;   // amount of padding around the grid

    private JComboBox  myComboBox;
    private JButton    myClearButton;
    private JButton    mySaveButton;
    private JButton    myQuitButton;
    private JButton    classifyVectorButton;

    private JLabel[][] myColorBoxes;   // an array of click-to-color labels
    private JLabel     resultLabel;
    private boolean    myPenOn; 
    private Color      myCurrentColor; // to keep track of the current color
    private String     digit; //what digit am I drawing?
    private boolean    digitSelected = false;

    // Stuff from ClassifierWindow before merge
    private static final int NUM_OUTPUT_CLASSES = 10;
    private static final int INPUT_VECTOR_DIMENSION = 256;  // The number of input units, not counting the bias unit.
    private static final int HIDDEN_LAYER_SIZE = 256;
    private static final String BORDER = new String("      ");
    private static final double epsilon = 1.0;
    private static final long DEFAULT_SEED = 478978392;
    private static final double DEFAULT_LAMBDA_VALUE = 1.0;
    private static final double DEFAULT_ALPHA = 0.001;
    private static final int DEFAULT_NUM_ITERATIONS = 5000000;
    private static final double STOP_THRESHOLD = 0.0001;
    // This stop the program if we grow too far above our achieved minimum
    private static final double GROWTH_THRESHOLD = 5.0;
    private static final double GRADIENT_CHECKING_EPSILON = 0.0001;
    private static final int MAX_DIMENSION_GRADIENT_CHECKING = 10;

    private JButton trainNetworkButton;
    private JButton saveThetasButton;
    private JButton readMatricesButton;
    private JButton classifyFromInputFileButton;

    private Random generator;
    private Matrix[] theta;
    private Matrix[] training;
    private Matrix[] output;
    private double lambda;
    private double alpha;
    private int numIterations;

    private DecimalFormat decimalFormat;
    public ClassifierWindow() {
        super( "Digit Classifier", 780, 800 );
        // use a border (geographic) layout 
        setLayout( new BorderLayout() );

        String[] options = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" };

        myComboBox = new JComboBox( options );
        myComboBox.setMaximumRowCount( options.length );
        myComboBox.addActionListener( this );

        JPanel comboPanel = new JPanel();
        comboPanel.add( myComboBox );
        add( comboPanel, BorderLayout.NORTH );

        add(comboPanel, BorderLayout.NORTH);
        JPanel southPanel = new JPanel();
        southPanel.setLayout(new GridLayout(2,1));

        add(southPanel, BorderLayout.SOUTH);
        JPanel eastPanel = new JPanel();
        eastPanel.setLayout(new GridLayout(3,1));
        eastPanel.add(new JLabel("                 "));
        resultLabel = new JLabel("Classified as:   ");
        eastPanel.add(resultLabel);
        eastPanel.add(new JLabel("                 "));
        add(eastPanel, BorderLayout.EAST);
        JPanel westPanel = new JPanel();
        JLabel westBorder = new JLabel(BORDER);
        westPanel.add(westBorder);
        add(westPanel, BorderLayout.WEST);

        JPanel topButtonPanel = new JPanel();

        myClearButton = new JButton( "Clear" );
        myClearButton.setActionCommand("clear");
        myClearButton.addActionListener( this );    // make it listen for mouse clicks
        //        myClearButton.addMouseListener( this );    // make it listen for mouse clicks

        mySaveButton = new JButton( "Save Image Vector" );
        mySaveButton.setActionCommand("save image vector");
        mySaveButton.addActionListener( this );    // make it listen for mouse clicks
        //        mySaveButton.addMouseListener( this );    // make it listen for mouse clicks

        classifyVectorButton = new JButton("Classify Vector");
        classifyVectorButton.setActionCommand("classify vector");
        classifyVectorButton.addActionListener( this);

        myQuitButton = new JButton( "Quit" );
        myQuitButton.setActionCommand("quit");
        myQuitButton.addActionListener( this );    // make it listen for mouse clicks
        //        myQuitButton.addMouseListener( this );    // make it listen for mouse clicks

        topButtonPanel.add( myClearButton ); 
        topButtonPanel.add( mySaveButton );
        topButtonPanel.add( classifyVectorButton );
        topButtonPanel.add( myQuitButton );

        southPanel.add(topButtonPanel);

        JPanel bottomButtonPanel = new JPanel();
        trainNetworkButton = new JButton("Train Network");
        trainNetworkButton.setActionCommand("trainNetwork");
        trainNetworkButton.addActionListener(this);
        bottomButtonPanel.add(trainNetworkButton);

        saveThetasButton = new JButton("Save Thetas");
        saveThetasButton.setActionCommand("saveThetas");
        saveThetasButton.addActionListener(this);
        bottomButtonPanel.add(saveThetasButton);

        readMatricesButton = new JButton("Read Matrices");
        readMatricesButton.setActionCommand("read matrices");
        readMatricesButton.addActionListener(this);
        bottomButtonPanel.add(readMatricesButton);

        classifyFromInputFileButton = new JButton("Classify From File");
        classifyFromInputFileButton.setActionCommand("classify_from_file");
        classifyFromInputFileButton.addActionListener(this);
        bottomButtonPanel.add(classifyFromInputFileButton);

        southPanel.add(bottomButtonPanel);

        add(southPanel, BorderLayout.SOUTH);

        JPanel gridPanel = new JPanel();
        // set its layout to be a 7X7 grid and draw an empty padding border around
        gridPanel.setLayout( new GridLayout( NUM_ROWS, NUM_COLS ) );
        gridPanel.setBorder( BorderFactory.createEmptyBorder( 10,25,0,0 ) );

        // construct the array
        myColorBoxes = new JLabel[NUM_ROWS][NUM_COLS];

        // then walk through the array one-by-one...
        for (int i = 0; i < myColorBoxes.length; i++)
        {
            for (int j = 0; j < myColorBoxes[i].length; j++)
            {
                // and construct each JLabel in the array
                myColorBoxes[i][j] = new JLabel();

                // add it to the panel
                gridPanel.add( myColorBoxes[i][j] );

                // make that background color show up (opaque becomes true)
                myColorBoxes[i][j].setBackground( null );
                myColorBoxes[i][j].setOpaque( true );

                // make each JLabel listen for mouse clicks
                myColorBoxes[i][j].addMouseListener( this );
            }
        }

        // add the panel of JLabels to the center of the window
        add( gridPanel, BorderLayout.CENTER );

        myPenOn = false;

        generator = new Random(DEFAULT_SEED);
        /* In our notes, the weight matrices are called theta1 and theta2.  So I do the same here for
         * consistency.  What that means is that theta[0] remains null, and that you use theta[1] and theta[2].
         */

        theta = new Matrix[3];
        lambda = DEFAULT_LAMBDA_VALUE;
        alpha = DEFAULT_ALPHA;
        numIterations = DEFAULT_NUM_ITERATIONS;
        decimalFormat = new DecimalFormat("#####0.###############");    

    }    

    public void buttonClicked(JButton whichButton) {
        String actionCommand = whichButton.getActionCommand();
        if (actionCommand.equals(trainNetworkButton.getActionCommand())) {
            
            trainMatrix();

        } else if (actionCommand.equals(saveThetasButton.getActionCommand())) {

            JFileChooser chooser = new JFileChooser( new File(".") );
            int value = chooser.showSaveDialog( this );
            if (value == JFileChooser.APPROVE_OPTION)
            {
                File file = chooser.getSelectedFile();
                try
                {
                    PrintWriter outputFile = new PrintWriter( file);

                    theta[1].print(outputFile, decimalFormat, 22);
                    outputFile.write("\n\n");
                    theta[2].print(outputFile, decimalFormat, 22);

                    outputFile.close();
                }
                catch (FileNotFoundException e)
                {
                    e.printStackTrace();
                }
                catch (IOException e) 
                {
                    e.printStackTrace();
                }
            }

        } else if (actionCommand.equals(readMatricesButton.getActionCommand())) {
            JFileChooser chooser = new JFileChooser( new File(".") );
            int value = chooser.showOpenDialog( this );
            if (value == JFileChooser.APPROVE_OPTION)
            {
                File file = chooser.getSelectedFile();

                try
                {
                    BufferedReader infile = new BufferedReader( new FileReader( file ) );
                    theta[1] = Matrix.read(infile);
                    theta[2] = Matrix.read(infile);
                    infile.close();
                } catch (FileNotFoundException e) {

                    e.printStackTrace();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }

        } else if ( actionCommand.equals(myClearButton.getActionCommand())) {

            clearImage();
        } else if ( actionCommand.equals(mySaveButton.getActionCommand())) {

            saveImage();
        }
        else if ( actionCommand.equals(myQuitButton.getActionCommand())) {

            System.exit(0);
        } else if (actionCommand.equals( classifyVectorButton.getActionCommand())) {
            String imageVector = "";

            for (int i = 0; i < myColorBoxes.length; i++)
            {
                for (int j = 0; j < myColorBoxes[i].length; j++)
                {
                    Color color = myColorBoxes[i][j].getBackground();
                    imageVector = imageVector + getColorChar(color);
                }

            }

            Matrix inputMatrix = inputStringToMatrix( imageVector );
            Matrix resultMatrix = computeHypothesis( inputMatrix, theta[1], theta[2] );

            int classifiedOutput = getMax( resultMatrix );

            resultLabel.setText("Classified as:   " + classifiedOutput);
            System.out.print("classification completed\n");
            System.out.flush();
        } else if (actionCommand.equals(classifyFromInputFileButton.getActionCommand())) {

            JFileChooser chooser = new JFileChooser( new File(".") );
            int value = chooser.showOpenDialog( this );
            int numVectors;
            int countCorrect = 0;
            if (value == JFileChooser.APPROVE_OPTION) {

                File file = chooser.getSelectedFile();

                try {
                    Scanner scanner = new Scanner(file);
                    String line;
                    Scanner parseLine = null;

                    String inputValue;
                    String outputValue;

                    Matrix inputVector;

                    // determine how many training vectors are in the file. I do this by counting instances of the colon char.
                    numVectors = 0;

                    while (scanner.hasNextLine()) {
                        line = scanner.nextLine();
                        if (line.contains(":")) {
                            ++numVectors;
                        }
                    }

                    // At this point, numTrainingVectors has the true number of training vectors.

                    // reset the scanner so it is at the beginning of the file
                    scanner.close();
                    scanner = new Scanner(file);

                    Matrix resultMatrix;
                    int classifiedOutput;
                    int count = 0;

                    while (scanner.hasNext()) {
                        line = scanner.next().trim();
                        if ((line.length() == 0) || (line.startsWith("#"))) {
                            continue;
                        }

                        parseLine = new Scanner( line );
                        parseLine.useDelimiter(":");

                        inputValue = parseLine.next().trim();
                        outputValue = parseLine.next().trim();

                        // this matrix already has a bias unit
                        inputVector = inputStringToMatrix(inputValue);

                        resultMatrix = computeHypothesis( inputVector, theta[1], theta[2] );

                        classifiedOutput = getMax( resultMatrix );
                        
                        if ( classifiedOutput == Integer.parseInt(outputValue)) {
                            ++countCorrect;
                        }

        
                    }     
                    System.out.print("\n" + countCorrect + " vectors out of " + numVectors + "classified correctly!\n");
                    int proportion = (int) (((countCorrect / (double) numVectors) * 100.0) + 0.5);
                    //            Properties properties = System.getProperties();
                    //            properties.list(System.out);
                    System.out.print("Percent correctly classified: " + proportion + "\n");
                    scanner.close();
                    System.out.print("classification completed\n");

                } catch (Exception e) {
                    e.printStackTrace();
                }
            }

        }
    }

    private void trainMatrix() {

        readTrainingData();
        
        /* Rather than just waste all the processing that goes into training a matrix,
         * this method saves the matrices (theta[1] and theta[2]) to a file, so they can be read in 
         * and used again.  I have provided the code that does the reading and writing of matrices.
         */

        // find out from the user which file they should use to save the matrices.

        JFileChooser chooser = new JFileChooser( new File(".") );
        int value = chooser.showSaveDialog( this );
        PrintWriter outputFile = null;

        if (value == JFileChooser.APPROVE_OPTION)
        {
            File file = chooser.getSelectedFile();
            try
            {
                outputFile = new PrintWriter( file);

            }
            catch (FileNotFoundException e)
            {
                e.printStackTrace();
            }
            catch (IOException e) 
            {
                e.printStackTrace();
            }
        }

        // So the first step in training the matrix is performing back propagation.
        theta = performBackPropagation();

        theta[1].print(outputFile, decimalFormat, 22);
        outputFile.write("\n\n");
        theta[2].print(outputFile, decimalFormat, 22);

        outputFile.close();

    }

    
    /* This method assumes that the input is a column vector (that is, a matrix with only a single
     * row.  It goes through the values in the matrix, and returns the ROW INDEX of the largest entry in the
     * matrix
     */
    
    private int getMax(Matrix m) {

        return 0;

    }

    /* 
     * This method assumes that the readTrainingData() method has been previously run, so
     * that the training and output arrays have already been filled.  If this assumption is not
     * valid, this method will likely throw a NullPointerException.  IMPORTANT: AN ASSUMPTION OF THIS
     * METHOD SHOULD BE THAT THE TRAINING VECTORS IN THE input ARRAY DO NOT HAVE BIAS UNITS.  You will
     * have to write code that adds that bias unit before you can perform back propagation with the
     * vectors.
     */

    private Matrix[] performBackPropagation() {
        // This neural network has only three layers, so only two theta matrices
        theta[1] = createInitialTheta(HIDDEN_LAYER_SIZE,INPUT_VECTOR_DIMENSION + 1);
        theta[2] = createInitialTheta(NUM_OUTPUT_CLASSES, HIDDEN_LAYER_SIZE + 1);

 
        return null;
    }


    private void readTrainingData() {
        JFileChooser chooser = new JFileChooser( new File(".") );
        int value = chooser.showOpenDialog( this );
        if (value == JFileChooser.APPROVE_OPTION) {

            File file = chooser.getSelectedFile();

            try {
                Scanner scanner = new Scanner(file);
                String line;
                Scanner parseLine = null;

                String inputValue;
                String outputValue;

                Matrix inputVector;
                Matrix yVector;

                // determine how many training vectors are in the file. I do this by counting instances of the colon char.
                int numTrainingVectors = 0;

                while (scanner.hasNextLine()) {
                    line = scanner.nextLine();
                    if (line.contains(":")) {
                        ++numTrainingVectors;
                    }
                }

                // At this point, numTrainingVectors has the true number of training vectors.

                // reset the scanner so it is at the beginning of the file
                scanner.close();
                scanner = new Scanner(file);

                training = new Matrix[numTrainingVectors];
                output = new Matrix[numTrainingVectors];
                int index = 0;

                while (scanner.hasNext()) {
                    line = scanner.next().trim();
                    if ((line.length() == 0) || (line.startsWith("#"))) {
                        continue;
                    }

                    parseLine = new Scanner( line );
                    parseLine.useDelimiter(":");

                    inputValue = parseLine.next().trim();
                    outputValue = parseLine.next().trim();
                    
                    // Make matrices out of these strings
                    /* One note: the method inputStringToMatrix() should NOT add the bias term to the input vector
                     * That is done by the method computeHypothesis().
                     */

                    training[index] = inputStringToMatrix(inputValue);
                    output[index] = vectorizeY(outputValue);
                    ++index;

                }
                

            } catch (Exception e) {
                e.printStackTrace();
            }
        }

    }

    /* This method should take as input a String representing a single digit (the correct digit) and creates the
     * correct output matrix for that digit.  So, for example, if the input String is "4", the output matrix should
     * be the 10 x 1 matrix
     * 
     *     0
     *     0
     *     0
     *     0
     *     1
     *     0
     *     0
     *     0
     *     0
     *     0
     *  
     */
    
    private static Matrix vectorizeY(String yValue) {

        return null;
    } 

    
    /* This method takes as input a String representing the binary representation of a digit.  Since the String should
     * have length INPUT_VECTOR_DIMENSION, one should end up with a matrix that has dimensions
     * INPUT_VECTROR_DIMENSION x 1.  Note that the 
     * 
     */
    private static Matrix inputStringToMatrix(String input) {

        return null;
    }

    
    /* This method takes as input the size (number of rows and number of cols) of a matrix, and creates a matrix
     * of the given size, which has random entries.  All entries of the matrix should fall between -epsilon and +epsilon,
     * where epsilon is the instance variable of the same name.
     */
    private Matrix createInitialTheta(int rows, int cols) {


        return null;

    }

    /* 
     * This method takes a double as input, and output the value of the logistic function when applied to x.
     */
    private double logisticFunction(double x) {

        return 0;

    }

    /* 
     * This method takes as input column vector, and creates a matrix whose entries are
     * the values of the logistic function performed on the entries of the input matrix.
     */
     
    private Matrix logisticFunction(Matrix x) {

        return null;
    }

    
    /* 
     * This method takes as input a single input vector (without bias unit -- you'll need to add that), along with the weight matrices, and
     * computes the output vector of the neural network. That is, it performs forward propagation.
     */
    private Matrix computeHypothesis(Matrix input, Matrix theta1, Matrix theta2) {
 
        return null;

    }

    /* This is a helper method.  It takes as input a matrix that results from the output of the neural network, and checks
     * that this vector is valid (all entries are between 0 and 1).  You may not need it.  I did.
     */
    
    private boolean validHypothesis(Matrix hypothesis) {

        // this method assumes a one dimensional column vector

        if (hypothesis.getColumnDimension() != 1) {
            System.out.print("Error: hypothesis not a column vector!\n");
            System.exit(1);
        }

        int length = hypothesis.getRowDimension();

        double entry;

        for (int row = 0; row < length; ++row) {
            entry = hypothesis.get(row, 0);
            if ( (entry <= 0) || (entry >= 1)) {
                System.out.print("Bad entry -> " + entry + "\n");
                return false;
            }            
        }

        return true;

    }

    /* This method takes as input an array of matrices that represent the input training data, an array of matrices that represent
     * the corresponding output data, an array of matrices that represent the weight matrices (well, the [1] and [2] index 
     * members do), and the value of lambda.  It returns an array of matrices, each of which represents some (but not all) of
     * the partial derivatives (with respect to the individual theta entries) of the theta matrices.  In particular, the
     * [1] index matrix of the output should correspond to the partials of theta[1].  The [2] index matrix of the output should
     * corresponds to the partials of theta[2].  When computing the partials, you should use GRADIENT_CHECKING_EPSILON
     * as the value of epsilon for gradient approximation purposes.
     */
    private Matrix[] gradientCheck(Matrix[] trainingData, Matrix[] outputData, Matrix[] thetaValues, double lambdaValue) {
        
        return null;
        
    }

    /* 
     * This method takes as input an array of matrices that represent the input training data, an array of matrices that represent
     * the corresponding output data, an array of matrices that represent the weight matrices (well, the [1] and [2] index 
     * members do), and the value of lambda.  It returns a double value that represents the value of the cost function J(theta) for
     * this choice of training data, theta values, and lambda.
     */
    
    private double jTheta(Matrix[] trainingData, Matrix[] outputData, Matrix[] thetaValues, double lambdaValue) {

        return 0;
        
    }

    /* You don't have to code this, but you might find it helpful for computing jTheta.  
     * It takes as input a matrix.  It computes the sum of the squares of each matrix entry,
     * with the exception of the first column of the matrix, which it ignores.
     */
    private double sumSquaredMatrixEntries(Matrix m) {
        
        return 0;
        
    }

    /* A helper method.  When debugging, it's sometimes convenient to be able to easily print out the dimensions of 
     * a matrix, along with a string that identifies to you which matrix this method is measuring.
     */
    private static void printMatrixDimensions(String name, Matrix m) {

        System.out.print("Matrix " + name + " has dimensions " + m.getRowDimension() + " x " + m.getColumnDimension() + "\n");

    }

    public void mousePressed( MouseEvent event )
    {
        myPenOn = true;   
    }

    public void mouseReleased( MouseEvent event )
    {
        myPenOn = false;
    }

    public void mouseEntered( MouseEvent e)
    {
        mouseExited(e);
    }

    public void mouseExited( MouseEvent event )
    {
        Component component     = event.getComponent();
        String    componentType = component.getClass().getName();
        // if the class type is a JLabel, call the abstract method 
        // to be implemented by the extending class 
        if ( myPenOn && componentType.contains( "JLabel" ) )
        {
            JLabel box = (JLabel) component;
            box.setBackground( BLACK );
        }        
    }

    //======================================================================
    //* public void labelClicked( JLabel whichLabel )
    //* This method is called whenever a JLabel is clicked.  Because only
    //* the labels in our grid are listening, this will only apply to those
    //* labels in the grid (and not the label at the bottom of the window).
    //======================================================================
    public void labelClicked( JLabel whichLabel )
    {
        // just set the label's color to the current color

        Color currentLabelBackgroundColor = whichLabel.getBackground();
        Color panelBackgroundColor = (whichLabel.getParent()).getBackground();

        if (currentLabelBackgroundColor.equals(BLACK)) {
            whichLabel.setBackground( panelBackgroundColor );
        } else {
            whichLabel.setBackground( BLACK );
        }
    }

    public void clearImage()
    {
        for (int i = 0; i < myColorBoxes.length; i++)
        {
            for (int j = 0; j < myColorBoxes[i].length; j++)
            {
                myColorBoxes[i][j].setBackground( null );
            }
        }
    }

    public char getColorChar( Color color )
    {
        if (color.equals( Color.black )) {
            return '1';
        }

        return '0';
    }

    public void saveImage()
    {
        if (!digitSelected) {

            Object[] possibilities = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
            String s = (String)JOptionPane.showInputDialog(
                    this,
                    "Please choose a digit!",
                    "Warning: No Digit Selected!",
                    JOptionPane.PLAIN_MESSAGE,
                    null,
                    possibilities,
                    "0");

            s = s.trim();
            if (s.length() != 1) {
                return;
            }

            char[] sAsChar = s.toCharArray();

            if (Character.isDigit(sAsChar[0])) {
                digit = s;
                digitSelected = true;
            } else {
                return;
            }

        }
        JFileChooser chooser = new JFileChooser( new File(".") );
        int value = chooser.showSaveDialog( this );
        if (value == JFileChooser.APPROVE_OPTION)
        {
            File file = chooser.getSelectedFile();
            try
            {
                PrintWriter outputFile = new PrintWriter( new BufferedWriter( new FileWriter( file.getName(), true )));
                String imageVector = "\n";

                for (int i = 0; i < myColorBoxes.length; i++)
                {
                    for (int j = 0; j < myColorBoxes[i].length; j++)
                    {
                        Color color = myColorBoxes[i][j].getBackground();
                        imageVector = imageVector + getColorChar(color);
                    }

                }
                imageVector = imageVector + ":" + digit;
                outputFile.write(imageVector);
                outputFile.write("\n\n");
                outputFile.close();
            }
            catch (FileNotFoundException e)
            {
                e.printStackTrace();
            }
            catch (IOException e) 
            {
                e.printStackTrace();
            }
        }
    }

    public void selectionMade( JComboBox whichComboBox )
    {
        digit = ((String) whichComboBox.getSelectedItem()).trim();
        digitSelected = true;
    }
}
