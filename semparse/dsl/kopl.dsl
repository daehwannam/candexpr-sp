;;; -*- mode: lisp -*-

;; Examples
;;
;; (defaction
;;   :name 'and
;;   :act_type 'bool
;;   :param_types '(bool bool bool)
;;   :expr_dict (mapkv :default $'(and_func @0 @1 @2)
;;                    :visual $'(and @0 @1 @2))
;;   :optional_idx None
;;   :rest_idx 2)
;;
;; (defaction
;;   :name 'or
;;   :act_type 'bool
;;   :param_types '(bool bool bool)
;;   :expr_dict (mapkv :default $'(or_func @0 @1 @2)
;;                    :visual $'(or @0 @1 @2))
;;   :optional_idx None
;;   :rest_idx 2)

(deftypes
  '(:object
    concept
    (:entity
     (:entity_with_fact
      entity_with_attr
      entity_with_rel))
    entity_name
    relation
    direction
    comp_operator
    st_or_bt_operator
    min_or_max_operator
    (:attribute
     attribute_string
     attribute_number
     (:attribute_time
      attribute_date
      attribute_year))
    (:qualifier
     qualifier_string
     qualifier_number
     (:qualifier_time
      qualifier_date
      qualifier_year))
    (:value
     value_string
     value_number
     (:value_time
      value_date
      value_year))))

(deftypes
  '(:result
    result_entity_name
    result_number
    result_relation
    result_attr_value
    result_attr_q_value
    result_rel_q_value
    result_boolean))


;; Result functions
;; 'Count',
;; 'QueryAttr',
;; 'QueryAttrQualifier',
;; 'QueryRelation',
;; 'QueryRelationQualifier',
;; 'SelectAmong',
;; 'SelectBetween',
;; 'VerifyDate',
;; 'VerifyNum',
;; 'VerifyStr',
;; 'VerifyYear',
;; QueryName ('What')


(defaction
  :name 'program
  :act_type 'result
  :param_types '(result)
  :expr_dict (mapkv :default $'(lambda (engine)
                                 (postprocess_denotation @0))
                    :visual "@0"
                    :visual_2 $'(program @0)))

(defaction
  :name 'all_entities
  :act_type 'entity
  :param_types '()
  :expr_dict (mapkv :default $'(engine.FindAll)
                    :visual $'all-entities))

(defaction
  :name 'find
  :act_type 'entity
  :param_types '(entity_name)
  :expr_dict (mapkv :default $'(engine.Find @0)
                    :visual $'(find @0)))

;;; filter_*

(defaction
  :name 'filter_concept
  :act_type 'entity
  :param_types '(concept)
  :expr_dict (mapkv :default $'(engine.FilterConcept @0)
                    :visual $'(filter-concept @0)))

(defaction
  :name 'filter_str
  :act_type 'entity_with_attr
  :param_types '(attribute_string value_string entity)
  :expr_dict (mapkv :default $'(engine.FilterStr @2 @0 @1)
                    :visual $'(filter-str @0 @1 @2)))

(defaction
  :name 'filter_num
  :act_type 'entity_with_attr
  :param_types '(attribute_number value_number comp_operator entity)
  :expr_dict (mapkv :default $'(engine.FilterNum @3 @0 @1 @2)
                    :visual $'(filter-num @0 @1 @2 @3)))

(defaction
  :name 'filter_year
  :act_type 'entity_with_attr
  :param_types '(attribute_time value_year comp_operator entity)
  :expr_dict (mapkv :default $'(engine.FilterYear @3 @0 @1 @2)
                    :visual $'(filter-year @0 @1 @2 @3)))

(defaction
  :name 'filter_date
  :act_type 'entity_with_attr
  :param_types '(attribute_time value_date comp_operator entity)
  :expr_dict (mapkv :default $'(engine.FilterDate @3 @0 @1 @2)
                    :visual $'(filter-date @0 @1 @2 @3)))

;;; relate

(defaction
  :name 'relate
  :act_type 'entity_with_rel
  :param_types '(relation direction entity)
  :expr_dict (mapkv :default $'(engine.Relate @2 @0 @1)
                    :visual $'(relate @0 @1 @2)))

;;; op_*

(defaction
  :name 'op_eq
  :act_type 'comp_operator
  :param_types '()
  :expr_dict (mapkv :default #"\"=\""
                    :visual "="))

(defaction
  :name 'op_ne
  :act_type 'comp_operator
  :param_types '()
  :expr_dict (mapkv :default #"\"!=\""
                    :visual "!="))

(defaction
  :name 'op_lt
  :act_type 'comp_operator
  :param_types '()
  :expr_dict (mapkv :default #"\"<\""
                    :visual "<"))

(defaction
  :name 'op_gt
  :act_type 'comp_operator
  :param_types '()
  :expr_dict (mapkv :default #"\">\""
                    :visual ">"))

;;; direction_*

(defaction
  :name 'direction_forward
  :act_type 'direction
  :param_types '()
  :expr_dict (mapkv :default #"\"forward\""
                    :visual "'forward"))

(defaction
  :name 'direction_backward
  :act_type 'direction
  :param_types '()
  :expr_dict (mapkv :default #"\"backward\""
                    :visual "'backward"))

;;; q_filter_*

(defaction
  :name 'q_filter_str
  :act_type 'entity_with_fact
  :param_types '(qualifier_string value_string entity_with_fact)
  :expr_dict (mapkv :default $'(engine.QFilterStr @2 @0 @1)
                    :visual $'(q-filter-str @0 @1 @2)))

(defaction
  :name 'q_filter_num
  :act_type 'entity_with_fact
  :param_types '(qualifier_number value_number comp_operator entity_with_fact)
  :expr_dict (mapkv :default $'(engine.QFilterNum @3 @0 @1 @2)
                    :visual $'(q-filter-num @0 @1 @2 @3)))

(defaction
  :name 'q_filter_year
  :act_type 'entity_with_fact
  :param_types '(qualifier_time value_year comp_operator entity_with_fact)
  :expr_dict (mapkv :default $'(engine.QFilterYear @3 @0 @1 @2)
                    :visual $'(q-filter-year @0 @1 @2 @3)))

(defaction
  :name 'q-filter_date
  :act_type 'entity_with_fact
  :param_types '(qualifier_time value_date comp_operator entity_with_fact)
  :expr_dict (mapkv :default $'(engine.QFilterDate @3 @0 @1 @2)
                    :visual $'(q-filter-date @0 @1 @2 @3)))

;;; intersect / union

(defaction
  :name 'intersect
  :act_type 'entity
  :param_types '(entity entity)
  :expr_dict (mapkv :default $'(engine.And @0 @1)
                    :visual $'(intersect @0 @1)))

(defaction
  :name 'union
  :act_type 'entity
  :param_types '(entity entity)
  :expr_dict (mapkv :default $'(engine.Or @0 @1)
                    :visual $'(union @0 @1)))

;;; count

(defaction
  :name 'count
  :act_type 'result_number
  :param_types '(entity)
  :expr_dict (mapkv :default $'(engine.Count @0)
                    :visual $'(count @0)))

;;; select_*

(defaction
  :name 'select_between
  :act_type 'result_entity_name
  :param_types '(attribute st_or_bt_operator entity entity)
  :expr_dict (mapkv :default $'(engine.SelectBetween @2 @3 @0 @1)
                    :visual $'(select-between @0 @1 @2 @3)))

(defaction
  :name 'select_among
  :act_type 'result_entity_name
  :param_types '(attribute min_or_max_operator entity)
  :expr_dict (mapkv :default $'(engine.SelectBetween @2 @0 @1)
                    :visual $'(select-between @0 @1 @2)))

;;; op_*

(defaction
  :name 'op_st
  :act_type 'st_or_bt_operator
  :param_types '()
  :expr_dict (mapkv :default #"\"smaller\""
                    :visual "'smaller"))

(defaction
  :name 'op_gt
  :act_type 'st_or_bt_operator
  :param_types '()
  :expr_dict (mapkv :default #"\"greater\""
                    :visual "'greater"))

(defaction
  :name 'op_min
  :act_type 'min_or_max_operator
  :param_types '()
  :expr_dict (mapkv :default #"\"min\""
                    :visual "'min"))

(defaction
  :name 'op_max
  :act_type 'min_or_max_operator
  :param_types '()
  :expr_dict (mapkv :default #"\"max\""
                    :visual "'max"))

;;; query_*

(defaction
  :name 'query_name
  :act_type 'result_entity_name
  :param_types '(entity)
  :expr_dict (mapkv :default $'(engine.QueryName @0)
                    :visual $'(query-name @0)))

(defaction
  :name 'query_attr
  :act_type 'result_attr_value
  :param_types '(attribute entity)
  :expr_dict (mapkv :default $'(engine.QueryAttr @1 @0)
                    :visual $'(query-attr @0 @1)))

(defaction
  :name 'query_attr_under_cond
  :act_type 'result_attr_value
  :param_types '(attribute qualifier value entity)
  :expr_dict (mapkv :default $'(engine.QueryAttrUnderCondition @3 @0 @1 @2)
                    :visual $'(query-attr-under-cond @0 @1 @2 @3)))

(defaction
  :name 'query_relation
  :act_type 'result_relation
  :param_types '(entity entity)
  :expr_dict (mapkv :default $'(engine.QueryRelation @0 @1)
                    :visual $'(query-relation @0 @1)))

(defaction
  :name 'query_attr_qualifier
  :act_type 'result_attr_q_value
  :param_types '(attribute value qualifier entity)
  :expr_dict (mapkv :default $'(engine.QueryAttrQualifier @3 @0 @1 @2)
                    :visual $'(query-attr-qualifier @0 @1 @2 @3)))

(defaction
  :name 'query_rel_qualifier
  :act_type 'result_rel_q_value
  :param_types '(relation qualifier entity entity)
  :expr_dict (mapkv :default $'(engine.QueryRelationQualifier @2 @3 @0 @1)
                    :visual $'(query-rel-qualifier @0 @1 @2 @3)))

;;; verify_*

(defaction
  :name 'verify_str
  :act_type 'result_boolean
  :param_types '(value_string result_attr_value)
  :expr_dict (mapkv :default $'(engine.VerifyStr @1 @0)
                    :visual $'(verify-str @0 @1)))

(defaction
  :name 'verify_num
  :act_type 'result_boolean
  :param_types '(value_number comp_operator result_attr_value)
  :expr_dict (mapkv :default $'(engine.VerifyNum @2 @0 @1)
                    :visual $'(verify-num @0 @1 @2)))

(defaction
  :name 'verify_year
  :act_type 'result_boolean
  :param_types '(value_year comp_operator result_attr_value)
  :expr_dict (mapkv :default $'(engine.VerifyYear @2 @0 @1)
                    :visual $'(verify-year @0 @1 @2)))

(defaction
  :name 'verify_date
  :act_type 'result_boolean
  :param_types '(value_date comp_operator result_attr_value)
  :expr_dict (mapkv :default $'(engine.VerifyDate @2 @0 @1)
                    :visual $'(verify-date @0 @1 @2)))

;;; parts


;; (defaction
;;   :name 'some-name
;;   :act_type 'some-type
;;   :param_types '(some-type some-type)
;;   :expr_dict (mapkv :default $'(some expression)
;;                    :visual $'(some expression))
;;   :optional_idx some-idx
;;   :rest_idx some-idx)
