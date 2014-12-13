package org.deeplearning4j;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.datasets.iterator.CSVDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.classifiers.dbn.DBN;
import org.deeplearning4j.nn.WeightInit;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.util.ImageLoader;
import org.deeplearning4j.util.SerializationUtils;
import org.nd4j.linalg.api.activation.Activations;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Collections;

/**
 * Hello world!
 *
 */
public class App  {

    private static Logger log = LoggerFactory.getLogger(App.class);

    public static void main( String[] args ) throws Exception  {
        DataSetIterator iter = new ImageDataSetIterator(4,4,28,28,new File("82"));
        RandomGenerator gen = new MersenneTwister(123);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().momentum(0.5)
                .momentumAfter(Collections.singletonMap(3, 0.9)).withActivationType(NeuralNetConfiguration.ActivationType.HIDDEN_LAYER_ACTIVATION)
                .iterations(1).weightInit(WeightInit.SIZE).applySparsity(true).sparsity(0.1)
                .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).rng(gen)
                .learningRate(1e-1f).nIn(784).nOut(2).list(4)
                .hiddenLayerSizes(new int[]{500, 250, 200})
                .override(new NeuralNetConfiguration.ConfOverride() {
                    @Override
                    public void override(int i, NeuralNetConfiguration.Builder builder) {
                        if (i == 3) {
                            builder.weightInit(WeightInit.ZERO);
                            builder.activationFunction(Activations.softMaxRows());
                            builder.lossFunction(LossFunctions.LossFunction.MCXENT);

                        }
                    }
                })
                .build();



        DBN d = new DBN.Builder().layerWiseConfiguration(conf)
                .build();


        while(iter.hasNext())
            d.fit(iter.next());


        iter.reset();

        Evaluation eval = new Evaluation();

        while(iter.hasNext()) {

            DataSet d2 = iter.next();
            INDArray predict2 = d.output(d2.getFeatureMatrix());

            eval.eval(d2.getLabels(), predict2);

        }

        log.info(eval.stats());



    }
}
