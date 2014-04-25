package org.plista.kornakapi.core.config;

import java.io.File;
import java.io.IOException;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.recommender.svd.FilePersistenceStrategy;
import org.apache.mahout.cf.taste.impl.recommender.svd.PersistenceStrategy;
import org.apache.mahout.cf.taste.model.DataModel;
import org.plista.kornakapi.KornakapiRecommender;
import org.plista.kornakapi.core.cluster.StreamingKMeansClassifierModel;
import org.plista.kornakapi.core.recommender.StreamingKMeansClassifierRecommender;
import org.plista.kornakapi.core.training.AbstractTrainer;
import org.plista.kornakapi.core.training.StreamingKMeansClustererTrainer;


public class StreamingKMeansClustererConfig extends RecommenderConfig{	
	
	private int desiredNumClusters;
	
	private long distanceCutoff;
	
	private long clusterTimeWindow;
	
	private StreamingKMeansClassifierModel model;
	
	
	public int getDesiredNumCluster(){
		return this.desiredNumClusters;
	}
	
	public long getDistanceCutoff(){
		return this.distanceCutoff;
	}
	
	public long getClusterTimeWindow(){
		return clusterTimeWindow;
	}
	public void setClusterTimeWindow(long clusterTimeWindow){
		this.clusterTimeWindow = clusterTimeWindow;
	}

	@Override
	public AbstractTrainer getTrainer()  {
		StreamingKMeansClustererTrainer clusterer;
		try {
			clusterer = new StreamingKMeansClustererTrainer( this, model);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		return clusterer;
	}

	@Override
	public void log() {
		log.info("Created StreamingKMeansClusterer [{}] with [{}] minclusters and [{}] cutoff distance",
	              new Object[] { this.getName(), this.getDesiredNumCluster(), this.getDistanceCutoff()}); 
	}

	@Override
	public KornakapiRecommender buildRecommenderFromConfig(Configuration conf,
			DataModel persistentData) throws IOException, TasteException {
		String name = this.getName();
  	  
        File modelFile = new File(conf.getModelDirectory(), name + ".model");

        PersistenceStrategy persistence = new FilePersistenceStrategy(modelFile);

        if (!modelFile.exists()) {
          createEmptyFactorization(persistence);
        }
        model = new StreamingKMeansClassifierModel(conf.getStorageConfiguration()); 
        StreamingKMeansClassifierRecommender recommender = new StreamingKMeansClassifierRecommender(model);
        return recommender;
	}
}
