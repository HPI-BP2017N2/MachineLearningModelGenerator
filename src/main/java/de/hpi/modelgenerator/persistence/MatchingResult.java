package de.hpi.modelgenerator.persistence;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
@JsonIgnoreProperties(ignoreUnknown = true)
public class MatchingResult {

    private long shopId;
    private String matchingReason;
    private int confidence;
    private String offerKey;
    private String idealoCategory;
    private String idealoCategoryName;
    private String higherLevelIdealoCategory;
    private String higherLevelIdealoCategoryName;
    private ParsedOffer parsedData;

}
