package de.hpi.machinelearning.persistence.persistence;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@RequiredArgsConstructor
@JsonIgnoreProperties(ignoreUnknown = true)
public class ScoredModel {

    private final byte[] modelByteArray;
    private final String modelType;
    private final double score;

}
