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

        //Integer folds= args[3]
        Integer folds=5;

        // Klasea esleitu
        data.setClassIndex(data.numAttributes() - 1);
        // Atributuak aukeratu filtroa aplikatuz
        AttributeSelection filter = applyAttributeSelection(data);
        // Datuei filtroa aplikatu
        Instances newData = Filter.useFilter(data, filter);
        // NaiveBayes classifier-a adierazi
        NaiveBayes classifier = new NaiveBayes();
        // 5-fold cross-validation
        Evaluation evaluator = crossValidateModel(classifier, newData, folds, new Random(1));
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

    private static void saveResults(Evaluation evaluator, String outputPath) throws Exception {
        FileWriter fileWriter = new FileWriter(outputPath);

        //Exekuzio data/ordua
        fileWriter.write("Data/ordua: " + java.time.LocalDateTime.now() + "\n\n");
        //Emaitza datuak
        fileWriter.write("Correctly Classified Instances: " + evaluator.pctCorrect() + " %\n");
        fileWriter.write("Incorrectly Classified Instances: " + evaluator.pctIncorrect() + " %\n" );
        fileWriter.write("Kappa: " + evaluator.kappa() + " %\n\n\n" );
        //Nahasmen matrizea
        fileWriter.write(evaluator.toMatrixString());
        //Precision metrikak
        fileWriter.write(evaluator.toClassDetailsString());
        fileWriter.write("\nWeighted Avg Precision:\n" + evaluator.weightedPrecision());
        //fileWriter Itxi
        fileWriter.flush();
        fileWriter.close();
    }
}
