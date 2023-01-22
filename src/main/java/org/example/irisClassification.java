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
    private static int FEATURES_COUNT = 4;
    private static int CLASSES_COUNT = 3;

    public static void main(String[] args) {


        BasicConfigurator.configure();
        loadData();

    }

    private static void saveBiasAndWeights(MultiLayerNetwork model) throws IOException {
        File locationToSave = new File("resources/savedModels/MyMultiLayerNetwork.zip");
        int i = 1;
        while (locationToSave.exists()){
            locationToSave = new File("resources/savedModels/MyMultiLayerNetwork_"+i+".zip");
            i++;
        }
        ModelSerializer.writeModel(model, locationToSave, true);
    }

    private static void irisNNetwork(DataSet trainingData, DataSet testData) throws IOException {
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(0.1, 0.91))
                .l2(0.0001)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(FEATURES_COUNT).nOut(3).build())
                .layer(1, new DenseLayer.Builder().nIn(3).nOut(3).build())
                .layer(2, new OutputLayer.Builder(
                        LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation(Activation.SOFTMAX)
                        .nIn(3).nOut(CLASSES_COUNT).build())
                .backprop(true).pretrain(false).build();

        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();
        model.fit(trainingData);

        INDArray output = model.output(testData.getFeatureMatrix());
        Evaluation eval = new Evaluation(3);
        eval.eval(testData.getLabels(), output);
        System.out.printf(eval.stats());

        saveBiasAndWeights(model);

    }


    private static void loadData() {
        try (RecordReader recordReader = new CSVRecordReader(0, ',')) {
            recordReader.initialize(new FileSplit(new ClassPathResource("iris.csv").getFile()
            ));

            DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, 150, 4, 3);
            DataSet allData = iterator.next();
            allData.shuffle(123);

            DataNormalization normalizer = new NormalizerStandardize();
            normalizer.fit(allData);
            normalizer.transform(allData);

            SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.80);
            DataSet trainingData = testAndTrain.getTrain();
            DataSet testingData = testAndTrain.getTest();

            irisNNetwork(trainingData,testingData);


        } catch (IOException | InterruptedException e) {
            throw new RuntimeException(e);
        }
    }
}