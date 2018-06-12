package de.hpi.modelgenerator.persistence.repo;

import de.hpi.machinelearning.persistence.ScoredModel;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;

import java.io.IOException;

public interface ModelRepository {

     void save(ParagraphVectors model, String type) throws IOException;

     void save(ScoredModel model) throws IOException;

     ParagraphVectors getBrandClassifier() throws IOException;

     boolean brandClassifierExists();
}
