package de.hpi.modelgenerator.persistence;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.index.Indexed;

@Getter
@Setter
@NoArgsConstructor
@JsonIgnoreProperties(ignoreUnknown = true)
public class MatchingResult {

    @Id private String url;
    @Indexed private String offerKey;
    private long shopId;
    private String matchingReason;
    private int confidence;
    private String idealoBrand;
    private String idealoCategory;
    private String idealoCategoryName;
    private String higherLevelIdealoCategory;
    private String higherLevelIdealoCategoryName;
    private ParsedOffer parsedData;

}
