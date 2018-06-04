package de.hpi.machinelearning.persistence;

import weka.core.Attribute;

import java.util.ArrayList;


public class AttributeVector extends ArrayList<Attribute> {

    public final static String TRUE = "true";
    public final static String FALSE = "false";
    public final static String NULL = "notSet";

    public AttributeVector() {
        Attribute jaccardTitle = new Attribute("jaccardSimilarityTitle");
        Attribute jaccardDescription = new Attribute("jaccardSimilarityDescription");
        Attribute cosineTitle = new Attribute("cosineSimilarityTitle");
        Attribute cosineDescription = new Attribute("cosineSimilarityDescription");
        Attribute priceDeviance = new Attribute("priceDeviance");

        ArrayList<String> booleans = new ArrayList<>();
        booleans.add(TRUE);
        booleans.add(FALSE);
        booleans.add(NULL);

        Attribute urlMatches = new Attribute("urlMatches", booleans);
        Attribute brandMatches = new Attribute("brandMatches", booleans);
        Attribute classifiedBrandMatches = new Attribute("classifiedBrandMatches", booleans);
        Attribute categoryMatches = new Attribute("categoryMatches", booleans);
        Attribute imageIdMatches = new Attribute("imageIdMatches", booleans);
        Attribute skuMatches = new Attribute("skuMatches", booleans);
        Attribute hanMatches = new Attribute("hanMatches", booleans);

        Attribute vectorClass = new Attribute("isMatch", booleans);

        this.add(jaccardTitle);
        this.add(cosineTitle);
        this.add(jaccardDescription);
        this.add(cosineDescription);
        this.add(priceDeviance);
        this.add(urlMatches);
        this.add(brandMatches);
        this.add(classifiedBrandMatches);
        this.add(categoryMatches);
        this.add(imageIdMatches);
        this.add(hanMatches);
        this.add(skuMatches);
        this.add(vectorClass);
    }
}