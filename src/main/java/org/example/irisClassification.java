package org.example;

import org.apache.log4j.BasicConfigurator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;

public class irisClassification {
    final private static int FEATURES_COUNT = 4;
    final private static int CLASSES_COUNT = 5;

    public static void main(String[] args) {
        //Current build takes Age, Sex(Male=0 Female=1), Race, Grip Strength
        //And tries to estimate Fearfulness score of the participant 1-5
        //Dataset used = https://www.kaggle.com/datasets/thedevastator/physical-strength-correlation-with-fear-related?resource=download&select=Sample_5_corrected.csv
        BasicConfigurator.configure();
        loadData();
    }

    private static void saveBiasAndWeights(MultiLayerNetwork model) throws IOException {
        File locationToSave = new File("resources/savedModels/MyMultiLayerNetwork.zip");
        int i = 1;
        while (locationToSave.exists()) {
            locationToSave = new File("resources/savedModels/MyMultiLayerNetwork_" + i + ".zip");
            i++;
        }
        ModelSerializer.writeModel(model, locationToSave, true);
    }

    private static void irisNNetwork(DataSet trainingData, DataSet testData, Boolean save) throws IOException {
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(0.35, 0.9))
                .l2(0.001)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(FEATURES_COUNT).nOut(5).build())
                .layer(1, new DenseLayer.Builder().nIn(5).nOut(4).build())
                .layer(2, new DenseLayer.Builder().nIn(4).nOut(3).build())
                .layer(3, new OutputLayer.Builder(
                        LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation(Activation.SOFTMAX)
                        .nIn(3).nOut(CLASSES_COUNT).build())
                .backprop(true).pretrain(false).build();

        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();
        model.fit(trainingData);

        INDArray output = model.output(testData.getFeatureMatrix());
        Evaluation eval = new Evaluation(5);
        eval.eval(testData.getLabels(), output);
        System.out.printf(eval.stats());
        System.out.println();
        System.out.println(eval.confusionToString());

        if (save) {
            saveBiasAndWeights(model);
        }
    }


    private static void loadData() {
        try (RecordReader recordReader = new CSVRecordReader(1, ',')) {
            recordReader.initialize(new FileSplit(new ClassPathResource("Physical Strength & Fear-Related Personality.csv").getFile()
            ));

            DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, 305, 4, 5);
            DataSet allData = iterator.next();
            allData.shuffle(123);

            DataNormalization normalizer = new NormalizerStandardize();
            normalizer.fit(allData);
            normalizer.transform(allData);

            SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.75);
            DataSet trainingData = testAndTrain.getTrain();
            DataSet testingData = testAndTrain.getTest();

            irisNNetwork(trainingData, testingData, true);


        } catch (IOException | InterruptedException e) {
            throw new RuntimeException(e);
        }
    }
}