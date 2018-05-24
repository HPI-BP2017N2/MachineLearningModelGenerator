package de.hpi.modelgenerator.api;

import de.hpi.modelgenerator.persistence.ScoredModel;
import de.hpi.modelgenerator.persistence.SerializedParagraphVectors;
import de.hpi.modelgenerator.persistence.repo.ModelRepository;
import de.hpi.modelgenerator.services.ModelGeneratorService;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

@RestController
@Slf4j
@Getter(AccessLevel.PRIVATE)
@RequiredArgsConstructor
public class ModelGeneratorController {

    private final ModelGeneratorService service;

    private final ModelRepository modelRepository;

    @RequestMapping(value = "/getCategoryClassifier/{shopId}", method = RequestMethod.GET, produces = "application/json")
    public SerializedParagraphVectors getCategoryClassifier(@PathVariable long shopId){
        /*SerializedParagraphVectors serializedNetwork = new SerializedParagraphVectors(getService().getCategoryClassifier(shopId));
        serializedNetwork.setNetworkType("category");
        getModelRepository().save(serializedNetwork);*/
        System.out.println(getModelRepository().loadModel("category"));
        //return serializedNetwork;
        return null;
    }

    @RequestMapping(value = "/getBrandClassifier/{shopId}", method = RequestMethod.GET, produces = "application/json")
    public SerializedParagraphVectors getBrandClassifier(@PathVariable long shopId){
        return new SerializedParagraphVectors(getService().getBrandClassifier(shopId));
    }

    @RequestMapping(value = "getModel", method = RequestMethod.GET, produces = "application/json")
    public ScoredModel getModel() {
        return getService().getModel();
    }
}
