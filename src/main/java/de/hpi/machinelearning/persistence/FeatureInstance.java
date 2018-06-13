package de.hpi.machinelearning.persistence;

import de.hpi.modelgenerator.persistence.ParsedOffer;
import de.hpi.modelgenerator.persistence.ShopOffer;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

import javax.validation.constraints.NotNull;
import java.util.ArrayList;
import java.util.Map;

import static de.hpi.machinelearning.TextSimilarityCalculator.cosineSimilarity;
import static de.hpi.machinelearning.TextSimilarityCalculator.jaccardSimilarity;

public class FeatureInstance extends DenseInstance {

    public FeatureInstance(@NotNull ShopOffer shopOffer, @NotNull ParsedOffer parsedOffer, @NotNull Boolean isMatch, String classifiedBrand) {
        super(13);
        final ArrayList<Attribute> features = new AttributeVector();
        Instances dataSet = new Instances("Rel", features, 1);
        dataSet.setClassIndex(12);
        this.setDataset(dataSet);
        this.setValue(features.get(0), jaccardSimilarity(getMapValue(shopOffer.getTitles()), parsedOffer.getTitle()));
        this.setValue(features.get(1), cosineSimilarity(getMapValue(shopOffer.getTitles()), parsedOffer.getTitle()));
        this.setValue(features.get(2), jaccardSimilarity(getMapValue(shopOffer.getDescriptions()), parsedOffer.getDescription())) ;
        this.setValue(features.get(3), cosineSimilarity(getMapValue(shopOffer.getDescriptions()), parsedOffer.getDescription()));
        this.setValue(features.get(4), getPercentualDeviance(getMapValue(shopOffer.getPrices()), parsedOffer.getPrice()));
        this.setValue(features.get(5), getEquation(getMapValue(shopOffer.getUrls()), parsedOffer.getUrl()));
        this.setValue(features.get(6), getEquation(shopOffer.getBrandName(), parsedOffer.getBrandName()));
        this.setValue(features.get(7), getEquation(shopOffer.getBrandName(), classifiedBrand));
        this.setValue(features.get(8), getEquation(shopOffer.getMappedCatalogCategory(), parsedOffer.getCategory()));
        this.setValue(features.get(9), compareImageIds(shopOffer.getImageId(), parsedOffer.getImageUrl()));
        this.setValue(features.get(10), getEquation(shopOffer.getHan(), parsedOffer.getHan()));
        this.setValue(features.get(11), getEquation(shopOffer.getSku(), parsedOffer.getSku()));
        this.setValue(features.get(12), String.valueOf(isMatch));
    }

    private <T> T getMapValue(Map<String, T> map) {
        return (map == null) ? null : map.get(map.keySet().iterator().next());
    }

    private String getEquation(String a, String b) {
        return (a == null || b == null) ? AttributeVector.NULL : a.toLowerCase().equals(b.toLowerCase())? AttributeVector.TRUE : AttributeVector.FALSE;
    }

    private double getPercentualDeviance(Double a, String b) {
        double bValue;

        if(a == null || b == null) {
            return -1;
        }

        try {
            bValue = Double.valueOf(b);
        } catch (NumberFormatException e) {
            return -1;
        }

        return (a == 0 || bValue == 0) ? -1 : 1 - (Math.min(a, bValue) / Math.max(a, bValue));
    }

    private String compareImageIds(String a, String b) {
        if(a == null || b == null) {
            return AttributeVector.NULL;
        }

        if(a.length() > b.length()) {
            return a.contains(b) ? AttributeVector.TRUE : AttributeVector.FALSE;
        } else if(b.length() > a.length()) {
            return b.contains(a) ? AttributeVector.TRUE : AttributeVector.FALSE;
        }

        return a.equals(b) ? AttributeVector.TRUE : AttributeVector.FALSE;
    }
}