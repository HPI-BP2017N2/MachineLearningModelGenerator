package de.hpi.modelgenerator.api;

import de.hpi.modelgenerator.services.ModelGeneratorService;
import de.hpi.modelgenerator.services.Classifier;
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

    @RequestMapping(value = "/trainCategory/{shopId}", method = RequestMethod.GET, produces = "application/json")
    public void trainCategory(@PathVariable long shopId){
        getService().classifyByCategory(shopId);
    }

    @RequestMapping(value = "/trainBrand/{shopId}", method = RequestMethod.GET, produces = "application/json")
    public void trainBrand(@PathVariable long shopId){
        getService().classifyByBrand(shopId);
    }

    @RequestMapping(value = "ml", method = RequestMethod.GET, produces = "application/json")
    public void ml() {
        getService().ml();
    }
}
