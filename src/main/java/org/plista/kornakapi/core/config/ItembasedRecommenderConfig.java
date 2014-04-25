/**
 * Copyright 2012 plista GmbH  (http://www.plista.com/)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and limitations under the License.
 */

package org.plista.kornakapi.core.config;

import java.io.File;
import java.io.IOException;

import org.apache.mahout.cf.taste.impl.recommender.AllSimilarItemsCandidateItemsStrategy;
import org.apache.mahout.cf.taste.impl.similarity.file.FileItemSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.plista.kornakapi.KornakapiRecommender;
import org.plista.kornakapi.core.recommender.ItemSimilarityBasedRecommender;
import org.plista.kornakapi.core.training.AbstractTrainer;
import org.plista.kornakapi.core.training.MultithreadedItembasedInMemoryTrainer;

/** configuration for recommenders that use item kNN */
public class ItembasedRecommenderConfig extends RecommenderConfig {

  private String similarityClass;
  private int similarItemsPerItem;

  public String getSimilarityClass() {
    return similarityClass;
  }

  public void setSimilarityClass(String similarityClass) {
    this.similarityClass = similarityClass;
  }

  public int getSimilarItemsPerItem() {
    return similarItemsPerItem;
  }

  public void setSimilarItemsPerItem(int similarItemsPerItem) {
    this.similarItemsPerItem = similarItemsPerItem;
  }
  
	protected File modelFile(Configuration conf, String recommenderName) {
		return new File(conf.getModelDirectory(), recommenderName + ".model");
	}

	@Override
	public KornakapiRecommender buildRecommenderFromConfig(Configuration conf,
			DataModel persistentData) throws IOException {
		String name = this.getName();
	
		File modelFile = modelFile(conf, name);
	
		if (!modelFile.exists()) {
			boolean created = modelFile.createNewFile();
			if (!created) {
				throw new IllegalStateException(
						"Cannot create file in model directory"
								+ conf.getModelDirectory());
			}
		}
	
		// set up recommender instance
		ItemSimilarity itemSimilarity = new FileItemSimilarity(
				modelFile);
		AllSimilarItemsCandidateItemsStrategy allSimilarItemsStrategy = new AllSimilarItemsCandidateItemsStrategy(
				itemSimilarity);
		KornakapiRecommender recommender = new ItemSimilarityBasedRecommender(
				persistentData, itemSimilarity,
				allSimilarItemsStrategy, allSimilarItemsStrategy);
		
		return recommender;
	}
	
	public AbstractTrainer getTrainer(){
		switch(trainer){
		case "MultithreadedItembasedInMemoryTrainer":
			return new MultithreadedItembasedInMemoryTrainer(this);
		}
		
		throw new IllegalArgumentException(trainer + " is unkown");
	}

	@Override
	public void log() {
		log.info(
				"Created ItemBasedRecommender [{}] using similarity [{}] and [{}] similar items per item",
				new Object[] { this.getName(),
						this.getSimilarityClass(),
						this.getSimilarItemsPerItem() });
	}
}
