package org.deeplearning4j;

import org.deeplearning4j.datasets.iterator.BaseDatasetIterator;

import java.io.File;

/**
 * Created by agibsonccc on 12/12/14.
 */
public class ImageDataSetIterator extends BaseDatasetIterator {
    public ImageDataSetIterator(int batch, int numExamples,int imageWidth,int imageHeight,File rootDir) {
        super(batch, numExamples, new ImageDataFetcher(imageWidth,imageHeight,rootDir));
    }
}
