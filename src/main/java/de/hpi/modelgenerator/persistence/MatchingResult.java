package de.hpi.modelgenerator.persistence;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@JsonIgnoreProperties(ignoreUnknown = true)
public class MatchingResult {

    private long shopId;
    private String matchingReason;
    private int confidence;
    private String offerKey;
    private ParsedOffer parsedData;

    public MatchingResult(long shopId,
                          String matchingReason,
                          int confidence,
                          String offerKey,
                          ParsedOffer parsedData) {
        setShopId(shopId);
        setMatchingReason(matchingReason);
        setConfidence(confidence);
        setOfferKey(offerKey);
        setParsedData(parsedData);
    }

}
