package de.hpi.modelgenerator.api;

import de.hpi.modelgenerator.services.ModelGeneratorService;
import de.hpi.modelgenerator.services.ParagraphVectorsClassifierExample;
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
    private final ParagraphVectorsClassifierExample categoryClassifier;

    @RequestMapping(value = "/train/{shopId}", method = RequestMethod.GET, produces = "application/json")
    public void train(@PathVariable long shopId){
        getCategoryClassifier().makeParagraphVectors(shopId);

        getCategoryClassifier().checkUnlabeledData(shopId);

    }

    @RequestMapping(value = "/label/{shopId}", method = RequestMethod.GET)
    public void label(@PathVariable long shopId){
        getCategoryClassifier().checkUnlabeledData(shopId);

    }


}
