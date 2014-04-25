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

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.recommender.svd.Factorization;
import org.apache.mahout.cf.taste.impl.recommender.svd.FilePersistenceStrategy;
import org.apache.mahout.cf.taste.impl.recommender.svd.PersistenceStrategy;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.CandidateItemsStrategy;
import org.plista.kornakapi.KornakapiRecommender;
import org.plista.kornakapi.core.recommender.CachingAllUnknownItemsCandidateItemsStrategy;
import org.plista.kornakapi.core.recommender.FoldingFactorizationBasedRecommender;
import org.plista.kornakapi.core.training.AbstractTrainer;
import org.plista.kornakapi.core.training.FactorizationbasedInMemoryTrainer;
import org.plista.kornakapi.core.training.MultithreadedItembasedInMemoryTrainer;

/** configuration for recommenders that use matrix factorization */
public class FactorizationbasedRecommenderConfig extends RecommenderConfig {

  private boolean usesImplicitFeedback;
  private int numberOfFeatures;
  private int numberOfIterations;
  private double lambda;
  private double alpha;

  public boolean isUsesImplicitFeedback() {
    return usesImplicitFeedback;
  }

  public void setUsesImplicitFeedback(boolean usesImplicitFeedback) {
    this.usesImplicitFeedback = usesImplicitFeedback;
  }

  public int getNumberOfFeatures() {
    return numberOfFeatures;
  }

  public void setNumberOfFeatures(int numberOfFeatures) {
    this.numberOfFeatures = numberOfFeatures;
  }

  public int getNumberOfIterations() {
    return numberOfIterations;
  }

  public void setNumberOfIterations(int numberOfIterations) {
    this.numberOfIterations = numberOfIterations;
  }

  public double getLambda() {
    return lambda;
  }

  public void setLambda(double lambda) {
    this.lambda = lambda;
  }

  public double getAlpha() {
    return alpha;
  }

  public void setAlpha(double alpha) {
    this.alpha = alpha;
  }
  


	@Override
	public KornakapiRecommender buildRecommenderFromConfig(Configuration conf,
			DataModel persistentData) throws IOException, TasteException {
		String name = this.getName();
	
		File modelFile = new File(conf.getModelDirectory(), name
				+ ".model");
	
		PersistenceStrategy persistence = new FilePersistenceStrategy(
				modelFile);
	
		if (!modelFile.exists()) {
			createEmptyFactorization(persistence);
		}
	
		CandidateItemsStrategy allUnknownItemsStrategy = new CachingAllUnknownItemsCandidateItemsStrategy(
				persistentData);
	
		FoldingFactorizationBasedRecommender svdRecommender = new FoldingFactorizationBasedRecommender(
				persistentData, allUnknownItemsStrategy, persistence);
		
		return svdRecommender;
	}

	public AbstractTrainer getTrainer(){
		switch(trainer){
		case "MultithreadedItembasedInMemoryTrainer":
			return new FactorizationbasedInMemoryTrainer(this);
		}
		
		throw new IllegalArgumentException(trainer + " is unkown");
	}
	
	@Override
	public void log() {
		log.info(
				"Created FactorizationBasedRecommender [{}] using [{}] features and [{}] iterations",
				new Object[] { this.getName(),
						this.getNumberOfFeatures(),
						this.getNumberOfIterations() });
	}
}
