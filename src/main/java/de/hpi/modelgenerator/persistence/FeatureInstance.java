package de.hpi.modelgenerator.persistence;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

import javax.validation.constraints.NotNull;
import java.util.ArrayList;
import java.util.Map;

import static de.hpi.modelgenerator.services.TextSimilarityCalculator.cosineSimilarity;
import static de.hpi.modelgenerator.services.TextSimilarityCalculator.jaccardSimilarity;

public class FeatureInstance extends DenseInstance {

    public FeatureInstance(@NotNull ShopOffer shopOffer, @NotNull ParsedOffer parsedOffer, @NotNull Boolean isMatch) {
        super(10);
        final ArrayList<Attribute> features = new AttributeVector();
        Instances dataSet = new Instances("Rel", features, 1);
        this.setDataset(dataSet);
        this.setValue(features.get(0), jaccardSimilarity(getMapValue(shopOffer.getTitles()), parsedOffer.getTitle()));
        this.setValue(features.get(1), cosineSimilarity(getMapValue(shopOffer.getTitles()), parsedOffer.getTitle()));
        this.setValue(features.get(2), jaccardSimilarity(getMapValue(shopOffer.getDescriptions()), parsedOffer.getDescription())) ;
        this.setValue(features.get(3), cosineSimilarity(getMapValue(shopOffer.getDescriptions()), parsedOffer.getDescription()));
        this.setValue(features.get(4), getPercentualDeviance(getMapValue(shopOffer.getPrices()), Double.valueOf(parsedOffer.getPrice())));
        this.setValue(features.get(5), getEquation(getMapValue(shopOffer.getUrls()), parsedOffer.getUrl()));
        this.setValue(features.get(6), getEquation(shopOffer.getBrandName(), parsedOffer.getBrandName()));
        this.setValue(features.get(7), getEquation(shopOffer.getMappedCatalogCategory(), parsedOffer.getCategory()));
        this.setValue(features.get(8), compareImageIds(shopOffer.getImageId(), parsedOffer.getImageUrl()));
        this.setValue(features.get(9), String.valueOf(isMatch));
    }

    private <T> T getMapValue(Map<String, T> map) {
        return (map == null)? null : map.get(map.keySet().iterator().next());
    }

    private String getEquation(String a, String b) {
        return (a == null || b == null)? null : a.toLowerCase().equals(b.toLowerCase())? "true" : "false";
    }

    private double getPercentualDeviance(double a, double b) {
        return 1 - (Math.min(a, b) / Math.max(a, b));
    }

    private String compareImageIds(String a, String b) {
        if(a == null || b == null) {
            return null;
        }

        if(a.length() > b.length()) {
            return a.contains(b)? "true" : "false";
        } else if(b.length() > a.length()) {
            return b.contains(a)? "true" : "false";
        }

        return a.equals(b)? "true" : "false";
    }
}
