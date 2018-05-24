package de.hpi.modelgenerator.persistence.repo;

import de.hpi.modelgenerator.persistence.ScoredModel;
import de.hpi.modelgenerator.persistence.SerializedParagraphVectors;

public interface ModelRepository {

    void save(SerializedParagraphVectors model);
    void save(ScoredModel model);
    boolean categoryClassifierExists();
    boolean brandClassifierExists();
    boolean modelExists();

}
