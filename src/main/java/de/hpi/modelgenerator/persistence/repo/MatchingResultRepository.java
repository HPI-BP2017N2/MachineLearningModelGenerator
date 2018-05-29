package de.hpi.modelgenerator.persistence.repo;

import de.hpi.modelgenerator.persistence.MatchingResult;

import java.util.List;
import java.util.Set;

public interface MatchingResultRepository {

    Set<Long> getShopIds();
    List<MatchingResult> getMatches(long shopId, int count);

}
