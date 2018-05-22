package de.hpi.modelgenerator.api;

import de.hpi.modelgenerator.dto.ScoredModel;
import de.hpi.modelgenerator.services.ModelGeneratorService;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
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

    @RequestMapping(value = "/getCategoryClassifier/{shopId}", method = RequestMethod.GET, produces = "application/json")
    public ParagraphVectors getCategoryClassifier(@PathVariable long shopId){
        return getService().getCategoryClassifier(shopId);
    }

    @RequestMapping(value = "/getBrandClassifier/{shopId}", method = RequestMethod.GET, produces = "application/json")
    public ParagraphVectors getBrandClassifier(@PathVariable long shopId){
        return getService().getBrandClassifier(shopId);
    }

    @RequestMapping(value = "getModel", method = RequestMethod.GET, produces = "application/json")
    public ScoredModel getModel() {
        return getService().getModel();
    }
}
