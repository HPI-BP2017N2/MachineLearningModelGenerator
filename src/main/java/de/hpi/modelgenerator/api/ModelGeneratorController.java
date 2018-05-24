package de.hpi.modelgenerator.api;

import de.hpi.modelgenerator.persistence.ClassifierTrainingState;
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

    private final ClassifierTrainingState categoryClassifierTrainingState = new ClassifierTrainingState();

    private final ClassifierTrainingState brandClassifierTrainingState = new ClassifierTrainingState();

    private final ClassifierTrainingState modelTrainigState = new ClassifierTrainingState();

    @RequestMapping(value = "/generateCategoryClassifier", method = RequestMethod.GET, produces = "application/json")
    public void generateCategoryClassifier() {
        if (!getModelRepository().categoryClassifierExists() && !getCategoryClassifierTrainingState().isCurrentlyLearning()) {
            getCategoryClassifierTrainingState().setCurrentlyLearning(true);
            getService().generateCategoryClassifier(288306L, getCategoryClassifierTrainingState());
        }
    }

    @RequestMapping(value = "/generateBrandClassifier", method = RequestMethod.GET, produces = "application/json")
    public void generateBrandClassifier() {
        if (!getModelRepository().brandClassifierExists() && !getBrandClassifierTrainingState().isCurrentlyLearning()) {
            getCategoryClassifierTrainingState().setCurrentlyLearning(true);
            getService().generateBrandClassifier(288306L, getBrandClassifierTrainingState());
        }
    }

    @RequestMapping(value = "generateModel", method = RequestMethod.GET, produces = "application/json")
    public void generateModel() {
        if (!getModelRepository().modelExists() && !getModelTrainigState().isCurrentlyLearning()) {
            getCategoryClassifierTrainingState().setCurrentlyLearning(true);
            getService().generateModel(getModelTrainigState());
        }
    }
}
