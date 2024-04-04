// Moore, Gemma
// AI - Multi Layer Perceptron - Boolean XOR 

import java.io.*;
import java.util.Scanner;
import java.text.NumberFormat;
import java.text.DecimalFormat;
import java.util.Random;
import java.lang.Math;

public class ExclusiveOr{

    public static double randomInRange(double start, double end) {
        Random r = new Random();
        return r.nextDouble() * (end - start) + start;
    }


    public static double sigmoid(double x){

        // calculate sigmoid
        return (1 / (1 + Math.exp(-x)));

    }
    public static void main(String []args) throws FileNotFoundException {

        // declare variables 
        // weights
        double w13 = randomInRange(-1.2, 1.2);
        double w14 = randomInRange(-1.2, 1.2);
        double w23 = randomInRange(-1.2, 1.2);
        double w24 = randomInRange(-1.2, 1.2);
        double w35 = randomInRange(-1.2, 1.2);
        double w45 = randomInRange(-1.2, 1.2);
        // theta 
        double theta3 = randomInRange(-1.2, 1.2);
        double theta4 = randomInRange(-1.2, 1.2);
        double theta5 = randomInRange(-1.2, 1.2);
        double  sse = -1;  // sum of square error        
        double [] Yd = new double[4];
        int [] x1 = new int[4];
        int [] x2 = new int[4];
        double [] Y3 = new double[4]; 
        double [] Y4 = new double[4]; 
        double [] Y5 = new double[4]; 
        double alpha = 0.1;
        double eGrad3, eGrad4, eGrad5, delt35, delt45, deltTheta5, delt13, delt23, deltTheta3, deltTheta4, delt14, delt24;    
        double[] e = {1, 1, 1, 1};
        int epoch = 0;

        // number formatter
        NumberFormat nf = DecimalFormat.getInstance();
        nf.setMaximumFractionDigits(4);  

        // read in training data from file

        // training data
        File trainingData = new File("trainingDataXOR.txt");  

        // scanner for traning data 
        Scanner scan = new Scanner(trainingData);    

        // implement training data using parallel arrays 
        // read in x1 as array 
        for(int i = 0; i < x1.length ; i++){
            x1[i] = scan.nextInt();
        }

        // read in x2 as array 
        for(int i = 0; i < x1.length; i++){
            x2[i] = scan.nextInt();
        }

        // read in Yd (desired output) as array 
        for(int i = 0; i < x1.length; i++){
            Yd[i] = scan.nextDouble();
        }

        // train 
        while (Math.abs(sse) > .001){ 

            for(int i = 0; i < Yd.length; i++){

                // activation 

                // calculate outputs of hidden layer 
                // neruons 3 and 4
                // Y3 
                Y3[i] = sigmoid(x1[i]*w13 + x2[i]*w23 - theta3);
                // Y4
                Y4[i] = sigmoid(x1[i]*w14 + x2[i]*w24 - theta4);

                // calculate outputs of output layer 
                // neuron 5
                Y5[i] = sigmoid(Y3[i]*w35 + Y4[i]*w45 - theta5);

                // calculate error
                e[i] = Yd[i] - Y5[i];        

                // weight training 
                eGrad5 = Y5[i]*(1-Y5[i])*e[i];
                delt35 = alpha * Y3[i]*eGrad5;
                delt45 = alpha * Y4[i]*eGrad5;
                deltTheta5 = alpha * -1 * eGrad5;

                eGrad3 = Y3[i]*(1-Y3[i])*eGrad5*w35;
                eGrad4 = Y4[i]*(1-Y4[i])*eGrad5*w45;

                // weight corrections
                delt13 = alpha * x1[i] * eGrad3;
                delt23 = alpha * x2[i] * eGrad3;
                deltTheta3 = alpha * (-1) * eGrad3;
                delt14 = alpha * x1[i] * eGrad4;
                delt24 = alpha * x2[i] * eGrad4;
                deltTheta4 = alpha * (-1) * eGrad4;

                // update all weights 
                w13 = w13 + delt13; 
                w14 = w14 + delt14;
                w23 = w23 + delt23;
                w24 = w24 + delt24;
                w35 = w35 + delt35;
                w45 = w45 + delt45;

                // update threshold levels
                theta3 = theta3 + deltTheta3;
                theta4 = theta4 + deltTheta4;
                theta5 = theta5 + deltTheta5;          
             
                

            }

            // calculate SSE
            sse = 0; 
            for (int i = 0; i < e.length; i++){
                sse += e[i] * e[i];
            }

            epoch++;

        }   

        // print info 
        System.out.println("# Epochs: " + epoch);
        System.out.println("x1\tx2\tYd\tY5\t\te");
        System.out.println("-----------------------------------------------");

        for(int i = 0; i<Yd.length; i++){
            System.out.println(x1[i] + "\t" + x2[i] + "\t" + Yd[i] + "\t" + nf.format(Y5[i]) + "\t\t" + nf.format(e[i]));

        }

        
    }
}