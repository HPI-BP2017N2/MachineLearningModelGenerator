package de.hpi.modelgenerator.persistence;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.index.Indexed;

import java.util.List;
import java.util.Map;

@Setter
@Getter
@NoArgsConstructor
@JsonIgnoreProperties(ignoreUnknown = true)
public class ShopOffer {

    @Id private String offerKey;
    @Indexed private byte phase = 0;
    private boolean isMatched = false;
    private Long shopId;
    private String brandName;
    private List<String> categoryPaths;
    private String productSearchtext;
    private String ean;
    private String han;
    private String sku;
    private Map<String, String> titles;
    private Map<String, Double> prices;
    private Map<String, String> descriptions;
    private Map<String, String> urls;
    private List<String> hans;
    private List<String> eans;
    private Map<String, String> smallPicture;
    private Map<String, List<String>> imageUrls;
    private String productKey;
    private String mappedCatalogCategory;

}
