package de.hpi.modelgenerator.persistence.repo;

import de.hpi.modelgenerator.persistence.SerializedParagraphVectors;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;

public interface ModelRepository {

    void save(SerializedParagraphVectors model);
    ParagraphVectors loadModel(String modelType);
}
