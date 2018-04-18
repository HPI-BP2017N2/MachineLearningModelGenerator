package de.hpi.modelgenerator.persistence;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.Getter;
import lombok.Setter;
import org.springframework.data.annotation.Id;

@Getter
@Setter
@JsonIgnoreProperties(ignoreUnknown = true)
public class ParsedOffer {

    @Id
    private String url;
    private long crawlingTimestamp;
    private String ean;
    private String sku;
    private String han;
    private String title;
    private String price;
    private String description;
    private String brandName;
    private String category;
}
