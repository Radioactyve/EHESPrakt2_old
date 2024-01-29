import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

import java.io.*;
import java.util.Random;

public class NaiveBayes5fCV {

    public static void main(String[] args) throws Exception {
        // Datuak kargatu
        // String inPath = args[0];
        String inPath = "C:/Users/radio/Desktop/System/Uni/3/Erabakiak/1. Praktika Datuak-20240122/heart-c.arff";
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(inPath);
        Instances data = source.getDataSet();
        // Emaitzak gorde
        // String outPath = args[1];
        String outPath = "C:/Users/radio/Desktop/emaitzakArik2.txt";



        // Klasea esleitu
        data.setClassIndex(data.numAttributes() - 1);
        // Atributuak aukeratu filtroa aplikatuz
        AttributeSelection filter = applyAttributeSelection(data); //(data) hartu jakiteko nolakoa izango den filtroa
        // Datuei filtroa aplikatu
        Instances newData = Filter.useFilter(data, filter);
        // NaiveBayes classifier-a adierazi
        NaiveBayes classifier = new NaiveBayes();

        //METODOAK
        // 5-fold cross-validation
        //Integer folds= args[3]
        Integer folds=5;
        Evaluation evaluator = crossValidateModel(classifier, newData, folds, new Random(1));
        // Hold-out
        Integer percent=60;
        Evaluation holdOutEvaluator = holdOutModel(classifier, newData, percent);


        // Emaitzak gorde
        saveResults(evaluator, outPath);
    }

    private static AttributeSelection applyAttributeSelection(Instances data) throws Exception {
        AttributeSelection filter = new AttributeSelection();
        CfsSubsetEval eval = new CfsSubsetEval();
        BestFirst search = new BestFirst();
        filter.setEvaluator(eval);
        filter.setSearch(search);
        filter.setInputFormat(data);
        return filter;
    }

    private static Evaluation crossValidateModel(NaiveBayes classifier, Instances data, int folds, Random seed)
            throws Exception {
        Evaluation evaluator = new Evaluation(data);
        evaluator.crossValidateModel(classifier, data, folds, seed);
        return evaluator;
    }
    private static Evaluation holdOutModel(NaiveBayes classifier, Instances data, double percentage) throws Exception {
        int numInstances = data.numInstances();

        //trainInstances lortu
        int numTrainInstances = (int) Math.round(numInstances * percentage / 100.0); //math round() zenbaki osoa lortzeko

        //train eta test instatziak sortu
        Instances trainData = new Instances(data, 0, numTrainInstances);
        Instances testData = new Instances(data, numTrainInstances, numInstances - numTrainInstances);

        //ebaluatu
        classifier.buildClassifier(trainData);
        Evaluation evaluator = new Evaluation(trainData);
        evaluator.evaluateModel(classifier, testData);

        return evaluator;
    }


    private static void saveResults(Evaluation evaluator, String outputPath) throws Exception {
        FileWriter fileWriter = new FileWriter(outputPath);
        //Exekuzio data/ordua
        fileWriter.write("Data/ordua: " + java.time.LocalDateTime.now() + "\n\n");

        //Emaitza datuak
        fileWriter.write("Correctly Classified Instances: " + evaluator.pctCorrect() + " %\n");
        fileWriter.write("Incorrectly Classified Instances: " + evaluator.pctIncorrect() + " %\n" );
        fileWriter.write("Kappa: " + evaluator.kappa() + " %\n" );
        fileWriter.write("MAE: " + evaluator.meanAbsoluteError() + " %\n" );
        fileWriter.write("RMSE: " + evaluator.rootMeanSquaredError() + " %\n" );
        fileWriter.write("RAE: " + evaluator.relativeAbsoluteError() + " %\n" );
        fileWriter.write("RRSE: " + evaluator.rootRelativeSquaredError() + " %\n\n\n" );

        //Nahasmen matrizea
        fileWriter.write(evaluator.toMatrixString());
        fileWriter.write("\n\n" );
        //Precision metrikak
        fileWriter.write(evaluator.toClassDetailsString());
        fileWriter.write("\nWeighted Avg Precision:" + evaluator.weightedPrecision());
        //fileWriter Itxi
        fileWriter.flush();
        fileWriter.close();
    }
}
