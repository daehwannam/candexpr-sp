
from collections import defaultdict, Counter

class KoPLRuntimeError(Exception):
    pass


def SelectBetween(engine, l_entities, r_entities, key, op):
    """
    Between two entities, query for entities with greater/smaller values for a specific property

    Args:
        l_entities (tuple): Two-tuple, the first member is a list of entities, the second member is None or a list of triples
        r_entities (tuple): Two-tuple, the first member is a list of entities, the second member is None or a list of triples
        key (string): attribute, requiring its attribute value to be a numeric type, such as "height"
        op (string): Comparator, "less" or "greater", which means querying entities with smaller or larger attribute values

    Returns:
        :obj:`str`: Returns the name of the entity

    """
    entity_ids_1, _ = l_entities
    entity_ids_2, _ = r_entities
    candidates = []
    for ent_id in entity_ids_1:
        for idx in engine.kb.attribute_inv_index[key][ent_id]:
            attr_info = engine.kb.entities[ent_id]['attributes'][idx]
            candidates.append((ent_id, attr_info['value']))
    for ent_id in entity_ids_2:
        for idx in engine.kb.attribute_inv_index[key][ent_id]:
            attr_info = engine.kb.entities[ent_id]['attributes'][idx]
            candidates.append((ent_id, attr_info['value']))
    candidates = list(filter(lambda x: x[1].type=='quantity', candidates))
    if len(candidates) == 0:
        raise KoPLRuntimeError('no candidate for `SelectBetween`')
    unit_cnt = defaultdict(int)
    for x in candidates:
        unit_cnt[x[1].unit] += 1   
    common_unit = Counter(unit_cnt).most_common()[0][0]
    candidates = list(filter(lambda x: x[1].unit==common_unit, candidates))
    sort = sorted(candidates, key=lambda x: x[1])
    i = sort[0][0] if op=='less' else sort[-1][0]
    name = engine.kb.entities[i]['name']
    return name
