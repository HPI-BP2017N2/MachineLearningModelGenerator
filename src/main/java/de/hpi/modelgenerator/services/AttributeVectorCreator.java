package de.hpi.modelgenerator.services;

import lombok.AccessLevel;
import lombok.Getter;
import lombok.Setter;
import weka.core.Attribute;
import weka.core.FastVector;

import java.lang.reflect.Array;
import java.util.ArrayList;

@Getter(AccessLevel.PRIVATE)
@Setter(AccessLevel.PRIVATE)
public class AttributeVectorCreator {

    public static ArrayList<Attribute> createFastVector(){
        ArrayList<Attribute> fvWekaAttributes = new ArrayList<>();
        Attribute Attribute1 = new Attribute("firstNumeric");
        ArrayList<String> fvNominalVal = new ArrayList<>();
        fvNominalVal.add("danial");
        fvNominalVal.add("daniela");
        Attribute Attribute2 = new Attribute("aNominal", fvNominalVal);
        ArrayList<String> classes = new ArrayList<>();
        classes.add("true");
        classes.add("false");
        Attribute ClassAttribute = new Attribute("theClass", classes);
        fvWekaAttributes.add(Attribute1);
        fvWekaAttributes.add(Attribute2);
        fvWekaAttributes.add(ClassAttribute);
        return fvWekaAttributes;
    }
}
