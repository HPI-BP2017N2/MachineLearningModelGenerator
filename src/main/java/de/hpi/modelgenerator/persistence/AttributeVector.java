package de.hpi.modelgenerator.persistence;

import weka.core.Attribute;

import java.util.ArrayList;


public class AttributeVector extends ArrayList<Attribute> {

    public AttributeVector() {
        Attribute jaccardTitle = new Attribute("jaccardSimilarityTitle");
        Attribute jaccardDescription = new Attribute("jaccardSimilarityDescription");
        Attribute cosineTitle = new Attribute("cosineSimilarityTitle");
        Attribute cosineDescription = new Attribute("cosineSimilarityDescription");
        Attribute priceDeviance = new Attribute("priceDeviance");

        ArrayList<String> booleans = new ArrayList<>();
        booleans.add("true");
        booleans.add("false");

        Attribute urlMatches = new Attribute("urlMatches", booleans);
        Attribute brandMatches = new Attribute("brandMatches", booleans);
        Attribute categoryMatches = new Attribute("categoryMatches", booleans);
        Attribute imageIdMatches = new Attribute("imageIdMatches", booleans);

        Attribute vectorClass = new Attribute("isMatch", booleans);

        this.add(jaccardTitle);
        this.add(cosineTitle);
        this.add(jaccardDescription);
        this.add(cosineDescription);
        this.add(priceDeviance);
        this.add(urlMatches);
        this.add(brandMatches);
        this.add(categoryMatches);
        this.add(imageIdMatches);
        this.add(vectorClass);
    }
}
