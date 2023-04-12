;;; -*- mode: lisp -*-

;; Examples
;;
;; (define-action
;;   :name 'and
;;   :act-type 'bool
;;   :param-types '(bool bool bool)
;;   :expr-dict (mapkv :default $(and-func @0 @1 @2)
;;                    :visual $(and @0 @1 @2))
;;   :optional-idx None
;;   :rest-idx 2)
;;
;; (define-action
;;   :name 'or
;;   :act-type 'bool
;;   :param-types '(bool bool bool)
;;   :expr-dict (mapkv :default $(or-func @0 @1 @2)
;;                    :visual $(or @0 @1 @2))
;;   :optional-idx None
;;   :rest-idx 2)

(define-types
  '(:object
    concept
    (:entity
     (:entity-with-fact
      entity-with-attr
      entity-with-rel))
    entity-name
    relation
    direction
    comp-operator
    st-or-bt-operator
    min-or-max-operator
    (:attribute
     attribute-string
     attribute-number
     (:attribute-time
      attribute-date
      attribute-year))
    (:qualifier
     qualifier-string
     qualifier-number
     (:qualifier-time
      qualifier-date
      qualifier-year))
    (:value
     value-string
     value-number
     value-quantity
     value-unit
     (:value-time
      value-date
      value-year))
    (:part
     part-string
     part-quantity
     part-unit
     (:part-time
      part-date
      part-year))))

(define-types
  '(:result
    result-entity-name
    result-number
    result-relation
    result-attr-value
    result-attr-q-value
    result-rel-q-value
    result-boolean))


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


(define-action
  :name 'program
  :act-type 'result
  :param-types '(result)
  :expr-dict (mapkv :default $(lambda (engine)
                                 (postprocess-denotation @0))
                    :visual "@0"
                    :visual-2 $(program @0)))

(define-action
  :name 'all-entities
  :act-type 'entity
  :param-types '()
  :expr-dict (mapkv :default $(engine.FindAll)
                    :visual "all-entities"))

(define-action
  :name 'find
  :act-type 'entity
  :param-types '(entity-name)
  :expr-dict (mapkv :default $(engine.Find @0)
                    :visual $(find @0)))

;;; filter-*

(define-action
  :name 'filter-concept
  :act-type 'entity
  :param-types '(concept)
  :expr-dict (mapkv :default $(engine.FilterConcept @0)
                    :visual $(filter-concept @0)))

(define-action
  :name 'filter-str
  :act-type 'entity-with-attr
  :param-types '(attribute-string value-string entity)
  :expr-dict (mapkv :default $(engine.FilterStr @2 @0 @1)
                    :visual $(filter-str @0 @1 @2)))

(define-action
  :name 'filter-number
  :act-type 'entity-with-attr
  :param-types '(attribute-number value-number comp-operator entity)
  :expr-dict (mapkv :default $(engine.FilterNum @3 @0 @1 @2)
                    :visual $(filter-number @0 @1 @2 @3)))

(define-action
  :name 'filter-year
  :act-type 'entity-with-attr
  :param-types '(attribute-time value-year comp-operator entity)
  :expr-dict (mapkv :default $(engine.FilterYear @3 @0 @1 @2)
                    :visual $(filter-year @0 @1 @2 @3)))

(define-action
  :name 'filter-date
  :act-type 'entity-with-attr
  :param-types '(attribute-time value-date comp-operator entity)
  :expr-dict (mapkv :default $(engine.FilterDate @3 @0 @1 @2)
                    :visual $(filter-date @0 @1 @2 @3)))

;;; relate

(define-action
  :name 'relate
  :act-type 'entity-with-rel
  :param-types '(relation direction entity)
  :expr-dict (mapkv :default $(engine.Relate @2 @0 @1)
                    :visual $(relate @0 @1 @2)))

;;; op-*

(define-action
  :name 'op-eq
  :act-type 'comp-operator
  :param-types '()
  :expr-dict (mapkv :default #"\"=\""
                    :visual "="))

(define-action
  :name 'op-ne
  :act-type 'comp-operator
  :param-types '()
  :expr-dict (mapkv :default #"\"!=\""
                    :visual "!="))

(define-action
  :name 'op-lt
  :act-type 'comp-operator
  :param-types '()
  :expr-dict (mapkv :default #"\"<\""
                    :visual "<"))

(define-action
  :name 'op-gt
  :act-type 'comp-operator
  :param-types '()
  :expr-dict (mapkv :default #"\">\""
                    :visual ">"))

;;; direction-*

(define-action
  :name 'direction-forward
  :act-type 'direction
  :param-types '()
  :expr-dict (mapkv :default #"\"forward\""
                    :visual "'forward"))

(define-action
  :name 'direction-backward
  :act-type 'direction
  :param-types '()
  :expr-dict (mapkv :default #"\"backward\""
                    :visual "'backward"))

;;; q-filter-*

(define-action
  :name 'q-filter-str
  :act-type 'entity-with-fact
  :param-types '(qualifier-string value-string entity-with-fact)
  :expr-dict (mapkv :default $(engine.QFilterStr @2 @0 @1)
                    :visual $(q-filter-str @0 @1 @2)))

(define-action
  :name 'q-filter-number
  :act-type 'entity-with-fact
  :param-types '(qualifier-number value-number comp-operator entity-with-fact)
  :expr-dict (mapkv :default $(engine.QFilterNum @3 @0 @1 @2)
                    :visual $(q-filter-number @0 @1 @2 @3)))

(define-action
  :name 'q-filter-year
  :act-type 'entity-with-fact
  :param-types '(qualifier-time value-year comp-operator entity-with-fact)
  :expr-dict (mapkv :default $(engine.QFilterYear @3 @0 @1 @2)
                    :visual $(q-filter-year @0 @1 @2 @3)))

(define-action
  :name 'q-filter-date
  :act-type 'entity-with-fact
  :param-types '(qualifier-time value-date comp-operator entity-with-fact)
  :expr-dict (mapkv :default $(engine.QFilterDate @3 @0 @1 @2)
                    :visual $(q-filter-date @0 @1 @2 @3)))

;;; intersect / union

(define-action
  :name 'intersect
  :act-type 'entity
  :param-types '(entity entity)
  :expr-dict (mapkv :default $(engine.And @0 @1)
                    :visual $(intersect @0 @1)))

(define-action
  :name 'union
  :act-type 'entity
  :param-types '(entity entity)
  :expr-dict (mapkv :default $(engine.Or @0 @1)
                    :visual $(union @0 @1)))

;;; count

(define-action
  :name 'count
  :act-type 'result-number
  :param-types '(entity)
  :expr-dict (mapkv :default $(engine.Count @0)
                    :visual $(count @0)))

;;; select-*

(define-action
  :name 'select-between
  :act-type 'result-entity-name
  :param-types '(attribute st-or-bt-operator entity entity)
  :expr-dict (mapkv :default $(engine.SelectBetween @2 @3 @0 @1)
                    :visual $(select-between @0 @1 @2 @3)))

(define-action
  :name 'select-among
  :act-type 'result-entity-name
  :param-types '(attribute min-or-max-operator entity)
  :expr-dict (mapkv :default $(engine.SelectBetween @2 @0 @1)
                    :visual $(select-between @0 @1 @2)))

;;; op-*

(define-action
  :name 'op-st
  :act-type 'st-or-bt-operator
  :param-types '()
  :expr-dict (mapkv :default #"\"smaller\""
                    :visual "'smaller"))

(define-action
  :name 'op-gt
  :act-type 'st-or-bt-operator
  :param-types '()
  :expr-dict (mapkv :default #"\"greater\""
                    :visual "'greater"))

(define-action
  :name 'op-min
  :act-type 'min-or-max-operator
  :param-types '()
  :expr-dict (mapkv :default #"\"min\""
                    :visual "'min"))

(define-action
  :name 'op-max
  :act-type 'min-or-max-operator
  :param-types '()
  :expr-dict (mapkv :default #"\"max\""
                    :visual "'max"))

;;; query-*

(define-action
  :name 'query-name
  :act-type 'result-entity-name
  :param-types '(entity)
  :expr-dict (mapkv :default $(engine.QueryName @0)
                    :visual $(query-name @0)))

(define-action
  :name 'query-attr
  :act-type 'result-attr-value
  :param-types '(attribute entity)
  :expr-dict (mapkv :default $(engine.QueryAttr @1 @0)
                    :visual $(query-attr @0 @1)))

(define-action
  :name 'query-attr-under-cond
  :act-type 'result-attr-value
  :param-types '(attribute qualifier value entity)
  :expr-dict (mapkv :default $(engine.QueryAttrUnderCondition @3 @0 @1 @2)
                    :visual $(query-attr-under-cond @0 @1 @2 @3)))

(define-action
  :name 'query-relation
  :act-type 'result-relation
  :param-types '(entity entity)
  :expr-dict (mapkv :default $(engine.QueryRelation @0 @1)
                    :visual $(query-relation @0 @1)))

(define-action
  :name 'query-attr-qualifier
  :act-type 'result-attr-q-value
  :param-types '(attribute value qualifier entity)
  :expr-dict (mapkv :default $(engine.QueryAttrQualifier @3 @0 @1 @2)
                    :visual $(query-attr-qualifier @0 @1 @2 @3)))

(define-action
  :name 'query-rel-qualifier
  :act-type 'result-rel-q-value
  :param-types '(relation qualifier entity entity)
  :expr-dict (mapkv :default $(engine.QueryRelationQualifier @2 @3 @0 @1)
                    :visual $(query-rel-qualifier @0 @1 @2 @3)))

;;; verify-*

(define-action
  :name 'verify-str
  :act-type 'result-boolean
  :param-types '(value-string result-attr-value)
  :expr-dict (mapkv :default $(engine.VerifyStr @1 @0)
                    :visual $(verify-str @0 @1)))

(define-action
  :name 'verify-number
  :act-type 'result-boolean
  :param-types '(value-number comp-operator result-attr-value)
  :expr-dict (mapkv :default $(engine.VerifyNum @2 @0 @1)
                    :visual $(verify-number @0 @1 @2)))

(define-action
  :name 'verify-year
  :act-type 'result-boolean
  :param-types '(value-year comp-operator result-attr-value)
  :expr-dict (mapkv :default $(engine.VerifyYear @2 @0 @1)
                    :visual $(verify-year @0 @1 @2)))

(define-action
  :name 'verify-date
  :act-type 'result-boolean
  :param-types '(value-date comp-operator result-attr-value)
  :expr-dict (mapkv :default $(engine.VerifyDate @2 @0 @1)
                    :visual $(verify-date @0 @1 @2)))

;;; constant-*

(define-action
  :name 'constant-string
  :act-type 'value-string
  :param-types '(part-string &rest part-string)
  :expr-dict (mapkv :default #(.format #"\"{}\"" (.lstrip (.replace (.join "" @:) "Ġ" " ")))))

(define-action
  :name 'constant-number
  :act-type 'value-number
  :param-types '(value-quantity value-unit)
  :expr-dict (mapkv :default #(.rstrip (.format #"\"{} {}\"" @1 @2))))

(define-action
  :name 'constant-quantity
  :act-type 'value-quantity
  :param-types '(part-quantity &rest part-quantity)
  :expr-dict (mapkv :default #(.format #"\"{}\"" (.lstrip (.replace (.join "" @:) "Ġ" " ")))))

(define-action
  :name 'constant-unit
  :act-type 'value-unit
  :param-types '(&rest part-unit)
  :expr-dict (mapkv :default #(.format #"\"{}\"" (.lstrip (.replace (.join "" @:) "Ġ" " ")))))

(define-action
  :name 'constant-date
  :act-type 'value-date
  :param-types '(part-date &rest part-date)
  :expr-dict (mapkv :default #(.format #"\"{}\"" (.lstrip (.replace (.join "" @:) "Ġ" " ")))))

(define-action
  :name 'constant-year
  :act-type 'value-year
  :param-types '(part-year &rest part-year)
  :expr-dict (mapkv :default #(.format #"\"{}\"" (.lstrip (.replace (.join "" @:) "Ġ" " ")))))

;;; token-*

(define-meta-action
  :meta-name 'token-string
  :name-fn (lambda (token) token)
  :expr-dict-fn (lambda (token)
                  (mapkv :default (.format #"\"{}\"" token)))
  :act-type 'part-string
  :param-types '())

(define-meta-action
  :meta-name 'token-quantity
  :name-fn (lambda (token) token)
  :expr-dict-fn (lambda (token)
                  (mapkv :default (.format #"\"{}\"" token)))
  :act-type 'part-quantity
  :param-types '())

(define-meta-action
  :meta-name 'token-unit
  :name-fn (lambda (token) token)
  :expr-dict-fn (lambda (token)
                  (mapkv :default (.format #"\"{}\"" token)))
  :act-type 'part-unit
  :param-types '())

(define-meta-action
  :meta-name 'token-date
  :name-fn (lambda (token) token)
  :expr-dict-fn (lambda (token)
                  (mapkv :default (.format #"\"{}\"" token)))
  :act-type 'part-date
  :param-types '())

(define-meta-action
  :meta-name 'token-year
  :name-fn (lambda (token) token)
  :expr-dict-fn (lambda (token)
                  (mapkv :default (.format #"\"{}\"" token)))
  :act-type 'part-year
  :param-types '())
