package org.deeplearning4j;

import java.io.File;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.base.LFWLoader;
import org.deeplearning4j.datasets.fetchers.BaseDataFetcher;
import org.deeplearning4j.util.ImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.util.FeatureUtil;


/**
 * Data fetcher for the LFW faces dataset
 * @author Adam Gibson
 *
 */
public class ImageDataFetcher extends BaseDataFetcher {

    /**
     *
     */
    private static final long serialVersionUID = -7473748140401804666L;
    private ImageLoader loader;
    private Iterator<File> fileIterator;
    private File rootDir;
    private List<String> labels;

    public ImageDataFetcher(int imageWidth,int imageHeight,File rootDir) {
        try {
            loader = new ImageLoader(imageWidth,imageHeight);
            this.rootDir = rootDir;
            inputColumns = imageWidth * imageHeight;
            if(!rootDir.isDirectory()) {
                throw new IllegalArgumentException("Illegal file: must be root directory");
            }
            labels = new ArrayList<>();
            for(String s : rootDir.list())
                 labels.add(s);
            numOutcomes = labels.size();
            fileIterator = FileUtils.iterateFiles(rootDir,null,true);
        } catch (Exception e) {
            throw new IllegalStateException("Unable to fetch images",e);
        }
    }


    public ImageDataFetcher(File rootDir) {
        this(200,200,rootDir);
    }


    @Override
    public boolean hasMore() {
        return fileIterator.hasNext();
    }

    @Override
    public void fetch(int numExamples) {
        if(!hasMore())
            throw new IllegalStateException("Unable to getFromOrigin more; there are no more images");



        //we need to ensure that we don't overshoot the number of examples total
        List<DataSet> toConvert = new ArrayList<>();

        for(int i = 0; i < numExamples; i++,cursor++) {
            if(!hasMore())
                break;
            try {
                File next = fileIterator.next();
                INDArray featureVector = loader.asRowVector(next);
                INDArray outcome = FeatureUtil.toOutcomeVector(labels.indexOf(next.getParentFile().getName()),labels.size());
                toConvert.add(new DataSet(featureVector,outcome));
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        initializeCurrFromList(toConvert);
    }




    @Override
    public DataSet next() {
        DataSet next = super.next();
        return next;
    }

    @Override
    public void reset() {
        fileIterator = FileUtils.iterateFiles(rootDir,null,true);
    }
}