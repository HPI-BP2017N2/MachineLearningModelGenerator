package de.hpi.modelgenerator.persistence.repo;

import de.hpi.machinelearning.persistence.SerializedParagraphVectors;
import de.hpi.machinelearning.persistence.persistence.ScoredModel;

public interface ModelRepository {

    void save(SerializedParagraphVectors model);
    void save(ScoredModel model);
    boolean categoryClassifierExists();
    boolean brandClassifierExists();
    boolean modelExists();

}
