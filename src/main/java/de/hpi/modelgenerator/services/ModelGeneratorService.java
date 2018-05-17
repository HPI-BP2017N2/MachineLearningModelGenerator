package de.hpi.modelgenerator.services;

import de.hpi.modelgenerator.persistence.ShopOffer;
import de.hpi.modelgenerator.persistence.repo.OfferRepository;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.springframework.stereotype.Service;

import java.util.LinkedList;
import java.util.List;

@Service
@RequiredArgsConstructor
@Getter
public class ModelGeneratorService {

    private final Classifier classifier = new Classifier();

    private final OfferRepository offerRepository;

    public void classifyByCategory(long shopId) {
        List<ShopOffer> offers = getOfferRepository().getOffers(shopId);
        List<LabelledDocument> documents = new LinkedList<>();

        for(ShopOffer offer : offers) {
            if(offer.getTitles() != null && offer.getMappedCatalogCategory() != null) {
                documents.add(getLabelledDocumentByCategoryFromShopOffer(offer));
            }
        }

        getClassifier().makeParagraphVectors(documents);
        getClassifier().checkUnlabeledData(shopId);

    }

    public void classifyByBrand(long shopId) {
        List<ShopOffer> offers = getOfferRepository().getOffers(shopId);
        List<LabelledDocument> documents = new LinkedList<>();

        for(ShopOffer offer : offers) {
            if(offer.getTitles() != null && offer.getMappedCatalogCategory() != null) {
                documents.add(getLabelledDocumentByBrandFromShopOffer(offer));
            }
        }

        getClassifier().makeParagraphVectors(documents);
        getClassifier().checkUnlabeledData(shopId);

    }

    public void ml(){
        WEKA weka =  new WEKA();
        weka.calculateModel();
        weka.evaluateModel();
    }

    private LabelledDocument getLabelledDocumentByCategoryFromShopOffer(ShopOffer offer) {
        LabelledDocument document = new LabelledDocument();
        String title = null;

        if(offer.getTitles() != null) {
            title = offer.getTitles().get(offer.getTitles().keySet().iterator().next());
        }

        document.setContent(title);
        document.addLabel(offer.getMappedCatalogCategory());

        return document;
    }

    private LabelledDocument getLabelledDocumentByBrandFromShopOffer(ShopOffer offer) {
        LabelledDocument document = new LabelledDocument();
        String title = null;

        if(offer.getTitles() != null) {
            title = offer.getTitles().get(offer.getTitles().keySet().iterator().next());
        }

        document.setContent(title);
        document.addLabel(offer.getBrandName());

        return document;
    }

}
