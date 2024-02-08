import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;

import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;

import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.supervised.instance.StratifiedRemoveFolds;

import java.util.Random;
import java.io.*;


public class NaiveBayes5fCV {

    public static void main(String[] args) throws Exception {
        // Datuak kargatu
        String inPath = args[0];
        DataSource source = new DataSource(inPath);
        Instances data = source.getDataSet();
        // Emaitzak gorde
        String outPath = args[1];

        // Klasea adierazi
        data.setClassIndex(data.numAttributes() - 1);
        // Atributuak aukeratu filtroa aplikatuz
        AttributeSelection filter = new AttributeSelection();
        filter.setEvaluator(new CfsSubsetEval());
        filter.setSearch(new BestFirst());
        filter.setInputFormat(data);

        // Datuei filtroa aplikatu
        Instances newData = Filter.useFilter(data, filter);
        // NaiveBayes classifier-a adierazi
        NaiveBayes classifier = new NaiveBayes();

        // ----------------------------------- [METODOAK] ---------------------------------------
        // 5-fold cross-validation
        if (args[2].equals("1")){
            Evaluation kfcv = new Evaluation(data);
            kfcv.crossValidateModel(classifier, newData, 5, new Random(1));
            saveResults(kfcv, outPath);
        }
        // Hold-out
        else if (args[2].equals("2")){
            Evaluation holdOut = holdOutModel(classifier, newData);
            saveResults(holdOut, outPath);
        }
        // Stratified Hold-out
        else{
            // Estratificar el conjunto de datos
            StratifiedRemoveFolds stratifiedFilter = new StratifiedRemoveFolds();
            stratifiedFilter.setNumFolds(2); // Dividir en dos conjuntos
            stratifiedFilter.setFold(1); // Seleccionar el primer conjunto (entrenamiento)
            stratifiedFilter.setInputFormat(newData);
            Instances trainData = Filter.useFilter(newData, stratifiedFilter);

            // Obtener el conjunto de datos restante (pruebas)
            stratifiedFilter.setFold(2); // Seleccionar el segundo conjunto (pruebas)
            Instances testData = Filter.useFilter(newData, stratifiedFilter);

            // Entrenar el clasificador
            classifier.buildClassifier(trainData);

            // Evaluar el clasificador en el conjunto de pruebas
            Evaluation sHoldOut = new Evaluation(trainData);
            sHoldOut.evaluateModel(classifier, testData);

            // Guardar los resultados en el archivo de salida
            saveResults(sHoldOut, outPath);
        }
    }

    private static Evaluation holdOutModel(NaiveBayes classifier, Instances data) throws Exception {
        //train eta test instatziak sortu
        int numInstances = data.numInstances();
        int numTrainInstances = (int) Math.round(numInstances * 0.66); //math round() zenbaki osoa lortzeko
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
        fileWriter.write(evaluator.toMatrixString() + "\n\n");
        //Precision metrikak
        fileWriter.write(evaluator.toClassDetailsString());
        fileWriter.write("\nWeighted Avg Precision:" + evaluator.weightedPrecision());

        //fileWriter itxi
        fileWriter.flush();
        fileWriter.close();
    }
}
