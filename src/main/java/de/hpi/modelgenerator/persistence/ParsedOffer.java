package de.hpi.modelgenerator.persistence;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.Getter;
import lombok.Setter;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.index.Indexed;

import java.util.Date;

@Getter
@Setter
@JsonIgnoreProperties(ignoreUnknown = true)
public class ParsedOffer {

    @Id private String url;
    private Date crawlingTimestamp;
    @Indexed
    private String ean;
    @Indexed private String han;
    private String imageUrl;
    private String sku;
    private String title;
    private String price;
    private String description;
    private String brandName;
    private String category;

}
